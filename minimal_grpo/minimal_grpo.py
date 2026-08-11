"""Minimal GRPO, written from scratch to understand the algorithm.

This is the learning artifact, not a trainer anyone should use at scale. It runs
on CPU with the same tiny Transformer that the repair tasks are built from, so
one model definition serves both purposes.

The algorithm, step by step:

  1. For each prompt, sample G completions from the current policy ("a group").
  2. Score every completion with a verifiable reward function.
  3. Turn rewards into advantages by normalizing WITHIN the group. This is the
     "group relative" part: the group mean acts as the baseline, which is why
     GRPO needs no learned value network.
  4. Recompute log-probs under the current policy and form the importance ratio
     against the log-probs from the sampling policy.
  5. Optimize the PPO clipped surrogate, plus a KL penalty to a frozen reference
     policy, averaged over *generated* tokens only.

Conventions follow the GRPO paper and the common open implementations
(open-thought/tiny-grpo, policy-gradient/GRPO-Zero, TRL's GRPOTrainer):
sequence-level advantage broadcast to every generated token, and Schulman's k3
estimator for the KL term.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow `python minimal_grpo/minimal_grpo.py` as well as `-m`.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from trgym.reference.tiny_gpt import TinyGPT, TinyGPTConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class GRPOConfig:
    group_size: int = 8           # G: completions sampled per prompt
    prompts_per_step: int = 4
    max_new_tokens: int = 12
    temperature: float = 1.0
    clip_eps: float = 0.2         # PPO clip range
    kl_beta: float = 0.02         # weight on the KL-to-reference penalty
    inner_epochs: int = 2         # reuse each batch of rollouts this many times
    # 3e-4 is the reflexive default and it does NOT move this task within 30
    # steps -- reward sits at the 6.25% random baseline. 1e-3 does. Recorded
    # because "the RL loop is correct but the step size is too small to see it"
    # is exactly the failure that wastes a paid training run.
    lr: float = 1e-3
    eos_token: int = 0
    seed: int = 0


@dataclass
class Rollouts:
    """One batch of sampled completions, flattened over (prompt, group)."""

    tokens: torch.Tensor          # (N, prompt_len + max_new) int64
    completion_mask: torch.Tensor  # (N, max_new) float, 1 = a real generated token
    old_logprobs: torch.Tensor    # (N, max_new) float, from the sampling policy
    rewards: torch.Tensor         # (N,) float
    prompt_len: int
    group_size: int
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Sampling
# --------------------------------------------------------------------------- #
def _token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target_t | prefix) for each position. logits: (N, T, V), targets: (N, T)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_group(
    policy: TinyGPT, prompts: torch.Tensor, cfg: GRPOConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `group_size` completions for every prompt.

    Returns (tokens, completion_mask, old_logprobs). Each prompt is repeated G
    times along the batch dimension, so row i belongs to prompt i // G.

    Deliberately recomputes the full forward pass every step instead of using a
    KV cache: correctness over speed, and the tasks in this repo are about
    training, not inference.
    """
    n_prompts, prompt_len = prompts.shape
    tokens = prompts.repeat_interleave(cfg.group_size, dim=0)  # (N, prompt_len)
    n = tokens.shape[0]

    logprobs = torch.zeros(n, cfg.max_new_tokens)
    mask = torch.zeros(n, cfg.max_new_tokens)
    alive = torch.ones(n, dtype=torch.bool)

    for step in range(cfg.max_new_tokens):
        logits = policy(tokens)[:, -1, :]                     # (N, V)
        probs = F.softmax(logits.float() / cfg.temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)         # (N, 1)

        # A finished sequence keeps emitting EOS, but those positions are masked
        # out of the loss, so they contribute nothing to the gradient.
        nxt = torch.where(alive.unsqueeze(1), nxt, torch.full_like(nxt, cfg.eos_token))

        logprobs[:, step] = torch.log(probs.gather(-1, nxt).squeeze(-1) + 1e-10)
        mask[:, step] = alive.float()

        tokens = torch.cat([tokens, nxt], dim=1)
        alive = alive & (nxt.squeeze(1) != cfg.eos_token)

    return tokens, mask, logprobs


# --------------------------------------------------------------------------- #
# 2-3. Rewards -> group-relative advantages
# --------------------------------------------------------------------------- #
def group_normalized_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-4
) -> torch.Tensor:
    """Normalize rewards within each group of `group_size` consecutive rows.

    A_i = (r_i - mean(group)) / (std(group) + eps)

    Two consequences worth internalizing:
      * The group mean is the baseline, so no value network is needed.
      * If every completion in a group earns the same reward the advantages are
        all zero and that group contributes no gradient. That is why a task with
        ~0% or ~100% pass rate teaches the policy nothing, and why environment
        difficulty has to be calibrated.
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


# --------------------------------------------------------------------------- #
# 4-5. The loss
# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over generated tokens only. Padding must never dilute the loss."""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def kl_k3(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimator of KL(policy || reference).

        k3 = exp(ref - new) - (ref - new) - 1

    Unbiased, always >= 0, and much lower variance than the naive (new - ref).
    """
    delta = ref_logprobs - new_logprobs
    return torch.exp(delta) - delta - 1.0


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """PPO clipped surrogate + KL penalty, averaged over generated tokens.

    `advantages` is per-sequence, shape (N,); it is broadcast to every generated
    token of that sequence.
    """
    adv = advantages.unsqueeze(1)                       # (N, 1) -> broadcast over T

    # Importance ratio between the current policy and the policy that sampled.
    # On the first inner epoch new == old, so ratio == 1 exactly and the
    # surrogate reduces to plain REINFORCE-with-baseline.
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

    # Minimum of the two terms => a pessimistic bound: the update is not allowed
    # to profit from moving the ratio far outside the trust region.
    policy_term = -torch.min(ratio * adv, clipped * adv)

    kl = kl_k3(new_logprobs, ref_logprobs)
    loss = masked_mean(policy_term, mask) + cfg.kl_beta * masked_mean(kl, mask)

    with torch.no_grad():
        clip_frac = masked_mean(
            ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).float(), mask
        )
        stats = {
            "kl": float(masked_mean(kl, mask)),
            "ratio_mean": float(masked_mean(ratio, mask)),
            "clip_frac": float(clip_frac),
            "policy_loss": float(masked_mean(policy_term, mask)),
        }
    return loss, stats


def sequence_entropy(policy: TinyGPT, tokens: torch.Tensor, prompt_len: int,
                     mask: torch.Tensor) -> float:
    """Mean per-token entropy over generated positions. Recorded, not optimized."""
    with torch.no_grad():
        logits = policy(tokens)[:, prompt_len - 1 : -1, :]
        logp = F.log_softmax(logits.float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
    return float(masked_mean(ent, mask))


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def compute_logprobs(policy: TinyGPT, tokens: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Log-probs of the generated tokens under `policy`.

    Position t of the logits predicts token t+1, so to score generated token at
    absolute index `prompt_len + j` we read logits at index `prompt_len + j - 1`.
    """
    logits = policy(tokens)[:, prompt_len - 1 : -1, :]
    targets = tokens[:, prompt_len:]
    return _token_logprobs(logits, targets)


def train(
    policy: TinyGPT,
    reward_fn,
    prompt_sampler,
    cfg: GRPOConfig,
    steps: int = 20,
    log: bool = True,
) -> list[dict]:
    """Run GRPO. `reward_fn(completions) -> (N,) float`, one score per rollout."""
    torch.manual_seed(cfg.seed)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history: list[dict] = []

    for step in range(steps):
        prompts = prompt_sampler(cfg.prompts_per_step)
        prompt_len = prompts.shape[1]

        policy.eval()
        tokens, mask, old_logprobs = sample_group(policy, prompts, cfg)
        completions = tokens[:, prompt_len:]
        rewards = reward_fn(completions, mask)
        advantages = group_normalized_advantages(rewards, cfg.group_size)

        with torch.no_grad():
            ref_logprobs = compute_logprobs(reference, tokens, prompt_len)

        policy.train()
        last_stats: dict = {}
        for _ in range(cfg.inner_epochs):
            new_logprobs = compute_logprobs(policy, tokens, prompt_len)
            loss, last_stats = grpo_loss(
                new_logprobs, old_logprobs, ref_logprobs, advantages, mask, cfg
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        record = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_absmax": float(advantages.abs().max()),
            "grad_norm": float(grad_norm),
            "entropy": sequence_entropy(policy, tokens, prompt_len, mask),
            "completion_len": float(mask.sum(dim=1).mean()),
            **last_stats,
        }
        history.append(record)
        if log:
            print(
                f"step {step:3d} | reward {record['reward_mean']:.3f} "
                f"| kl {record['kl']:.5f} | entropy {record['entropy']:.3f} "
                f"| clip_frac {record['clip_frac']:.3f} "
                f"| len {record['completion_len']:.1f}"
            )
    return history


# --------------------------------------------------------------------------- #
# A toy verifiable task, so the loop can actually be run end to end
# --------------------------------------------------------------------------- #
ALLOWED_TOKENS = torch.arange(64, 80)  # 16 of 256 ids => 6.25% random baseline


def allowed_token_reward(completions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fraction of generated tokens drawn from ALLOWED_TOKENS.

    Deliberately trivial and fully verifiable: the point is to exercise the RL
    machinery, not to be an interesting task.
    """
    hit = torch.isin(completions, ALLOWED_TOKENS).float()
    return (hit * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def make_prompt_sampler(cfg: GRPOConfig, prompt_len: int = 4, vocab_size: int = 256):
    def sample(n: int) -> torch.Tensor:
        return torch.randint(128, vocab_size, (n, prompt_len))

    return sample


def main() -> None:
    import json

    cfg = GRPOConfig()
    torch.manual_seed(cfg.seed)
    policy = TinyGPT(TinyGPTConfig())
    history = train(policy, allowed_token_reward, make_prompt_sampler(cfg), cfg, steps=60)

    out = Path(_REPO_ROOT) / "artifacts" / "grpo_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"config": cfg.__dict__, "history": history}, indent=2),
                   encoding="utf-8")
    first = sum(h["reward_mean"] for h in history[:5]) / 5
    last = sum(h["reward_mean"] for h in history[-5:]) / 5
    print(
        f"\nrandom baseline  {len(ALLOWED_TOKENS)}/256 = "
        f"{len(ALLOWED_TOKENS) / 256:.4f}"
        f"\nreward (mean of first 5 steps) {first:.4f}"
        f"\nreward (mean of last 5 steps)  {last:.4f}"
        f"\nwrote {out}"
    )


if __name__ == "__main__":
    main()
