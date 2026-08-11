"""Unit tests for the from-scratch GRPO implementation.

Rather than diffing against another repository (which would only prove the two
agree, not that either is right), each test pins a mathematical property that
any correct GRPO must satisfy. Where a convention is borrowed -- the k3 KL
estimator -- it is checked numerically against a Monte-Carlo estimate of the
same quantity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minimal_grpo.minimal_grpo import (
    ALLOWED_TOKENS,
    GRPOConfig,
    allowed_token_reward,
    compute_logprobs,
    group_normalized_advantages,
    grpo_loss,
    kl_k3,
    make_prompt_sampler,
    masked_mean,
    sample_group,
    train,
)
from trgym.reference.tiny_gpt import TinyGPT, TinyGPTConfig

CFG = GRPOConfig(group_size=4, prompts_per_step=2, max_new_tokens=6)


# --------------------------------------------------------------------------- #
# Advantages
# --------------------------------------------------------------------------- #
def test_advantages_are_zero_mean_unit_std_within_group() -> None:
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 10.5, 11.0, 11.5])
    adv = group_normalized_advantages(rewards, group_size=4)
    per_group = adv.view(-1, 4)
    assert torch.allclose(per_group.mean(dim=1), torch.zeros(2), atol=1e-3)
    assert torch.allclose(per_group.std(dim=1, unbiased=False), torch.ones(2), atol=1e-3)


def test_constant_reward_group_produces_zero_advantage() -> None:
    """No spread inside a group => no learning signal. This is the difficulty-
    calibration constraint expressed as an equation."""
    rewards = torch.tensor([0.7, 0.7, 0.7, 0.7])
    adv = group_normalized_advantages(rewards, group_size=4)
    assert torch.allclose(adv, torch.zeros(4), atol=1e-3)


def test_advantage_ordering_matches_reward_ordering() -> None:
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.5])
    adv = group_normalized_advantages(rewards, group_size=4)
    assert torch.equal(rewards.argsort(), adv.argsort())


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def test_masked_mean_ignores_padding() -> None:
    x = torch.tensor([[1.0, 2.0, 999.0], [3.0, 999.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert float(masked_mean(x, mask)) == 2.0  # (1 + 2 + 3) / 3


def test_masked_mean_handles_all_padding() -> None:
    x = torch.ones(2, 3)
    assert float(masked_mean(x, torch.zeros(2, 3))) == 0.0


# --------------------------------------------------------------------------- #
# KL
# --------------------------------------------------------------------------- #
def test_kl_k3_is_zero_when_policies_match() -> None:
    lp = torch.randn(4, 5)
    assert torch.allclose(kl_k3(lp, lp), torch.zeros(4, 5), atol=1e-7)


def test_kl_k3_is_non_negative() -> None:
    torch.manual_seed(0)
    new = torch.randn(64, 8)
    ref = torch.randn(64, 8)
    assert bool((kl_k3(new, ref) >= 0).all())


def test_kl_k3_matches_monte_carlo_kl() -> None:
    """E_{x~p}[k3] should approximate KL(p||q) for real distributions."""
    torch.manual_seed(1)
    vocab = 12
    p_logits = torch.randn(vocab)
    q_logits = p_logits + 0.3 * torch.randn(vocab)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)

    true_kl = float((log_p.exp() * (log_p - log_q)).sum())

    samples = torch.multinomial(log_p.exp(), num_samples=200_000, replacement=True)
    estimate = float(kl_k3(log_p[samples], log_q[samples]).mean())

    assert estimate == __import__("pytest").approx(true_kl, rel=0.05, abs=1e-4)


# --------------------------------------------------------------------------- #
# The surrogate
# --------------------------------------------------------------------------- #
def _loss_inputs(n: int = 8, t: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    old = torch.randn(n, t)
    adv = torch.randn(n)
    mask = torch.ones(n, t)
    return old, adv, mask


def test_ratio_is_one_on_first_inner_epoch() -> None:
    old, adv, mask = _loss_inputs()
    _, stats = grpo_loss(old.clone(), old, old, adv, mask, CFG)
    assert stats["ratio_mean"] == __import__("pytest").approx(1.0, abs=1e-6)
    assert stats["clip_frac"] == 0.0


def test_at_ratio_one_surrogate_equals_reinforce_with_baseline() -> None:
    """With new == old the clipped surrogate must reduce to -A * logprob."""
    old, adv, mask = _loss_inputs()
    new = old.clone().requires_grad_(True)
    cfg = GRPOConfig(**{**CFG.__dict__, "kl_beta": 0.0})

    loss, _ = grpo_loss(new, old, old, adv, mask, cfg)
    loss.backward()
    got = new.grad.clone()

    new2 = old.clone().requires_grad_(True)
    reinforce = masked_mean(-adv.unsqueeze(1) * new2, mask)
    reinforce.backward()
    assert torch.allclose(got, new2.grad, atol=1e-6)


def test_clipping_activates_outside_trust_region() -> None:
    old = torch.zeros(4, 3)
    new = torch.full((4, 3), 0.5)  # ratio = e^0.5 = 1.65 > 1 + 0.2
    adv = torch.ones(4)
    mask = torch.ones(4, 3)
    _, stats = grpo_loss(new, old, old, adv, mask, CFG)
    assert stats["clip_frac"] == 1.0


def test_clipped_positive_advantage_gradient_is_zero() -> None:
    """Beyond 1+eps with A > 0 the surrogate is flat: no incentive to push further."""
    old = torch.zeros(4, 3)
    new = torch.full((4, 3), 0.5, requires_grad=True)
    adv = torch.ones(4)
    mask = torch.ones(4, 3)
    cfg = GRPOConfig(**{**CFG.__dict__, "kl_beta": 0.0})
    loss, _ = grpo_loss(new, old, old, adv, mask, cfg)
    loss.backward()
    assert torch.allclose(new.grad, torch.zeros_like(new.grad), atol=1e-8)


def test_zero_advantage_gives_zero_policy_gradient() -> None:
    old = torch.zeros(4, 3)
    new = torch.zeros(4, 3, requires_grad=True)
    cfg = GRPOConfig(**{**CFG.__dict__, "kl_beta": 0.0})
    loss, _ = grpo_loss(new, old, old, torch.zeros(4), torch.ones(4, 3), cfg)
    loss.backward()
    assert torch.allclose(new.grad, torch.zeros_like(new.grad), atol=1e-8)


# --------------------------------------------------------------------------- #
# Log-prob alignment (the classic off-by-one in an RL trainer)
# --------------------------------------------------------------------------- #
def test_compute_logprobs_aligns_with_sampled_tokens() -> None:
    """Recomputed log-probs must match the ones recorded during sampling."""
    torch.manual_seed(0)
    policy = TinyGPT(TinyGPTConfig()).eval()
    cfg = GRPOConfig(group_size=2, prompts_per_step=2, max_new_tokens=5, temperature=1.0)
    prompts = torch.randint(0, 256, (2, 4))

    tokens, mask, old_logprobs = sample_group(policy, prompts, cfg)
    recomputed = compute_logprobs(policy, tokens, prompt_len=4)

    diff = ((recomputed.detach() - old_logprobs).abs() * mask).max()
    assert float(diff) < 1e-4, f"log-prob misalignment {float(diff)}"


def test_sampling_shapes_and_mask_semantics() -> None:
    torch.manual_seed(0)
    policy = TinyGPT(TinyGPTConfig()).eval()
    cfg = GRPOConfig(group_size=3, prompts_per_step=2, max_new_tokens=7)
    prompts = torch.randint(1, 256, (2, 4))
    tokens, mask, logprobs = sample_group(policy, prompts, cfg)

    assert tokens.shape == (6, 4 + 7)
    assert mask.shape == logprobs.shape == (6, 7)
    assert bool(((mask == 0) | (mask == 1)).all())
    # mask must be non-increasing: once a sequence stops it stays stopped
    assert bool((mask[:, 1:] <= mask[:, :-1]).all())
    assert bool((logprobs <= 0).all())


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
def test_reward_is_fraction_of_allowed_tokens() -> None:
    completions = torch.tensor([[64, 65, 200], [200, 201, 202]])
    mask = torch.ones(2, 3)
    r = allowed_token_reward(completions, mask)
    assert float(r[0]) == __import__("pytest").approx(2 / 3)
    assert float(r[1]) == 0.0


def test_reward_respects_mask() -> None:
    completions = torch.tensor([[64, 200, 200]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(allowed_token_reward(completions, mask)[0]) == 1.0


# --------------------------------------------------------------------------- #
# End to end: the loop must actually learn something
# --------------------------------------------------------------------------- #
def test_training_increases_reward_on_toy_task() -> None:
    cfg = GRPOConfig(
        group_size=8, prompts_per_step=4, max_new_tokens=8, lr=1e-3, inner_epochs=2, seed=0
    )
    torch.manual_seed(cfg.seed)
    policy = TinyGPT(TinyGPTConfig())
    history = train(policy, allowed_token_reward, make_prompt_sampler(cfg), cfg,
                    steps=25, log=False)

    first = sum(h["reward_mean"] for h in history[:5]) / 5
    last = sum(h["reward_mean"] for h in history[-5:]) / 5
    assert last > first + 0.05, f"reward did not improve: {first:.3f} -> {last:.3f}"
    assert all(h["kl"] >= 0 for h in history)
    assert all(h["grad_norm"] == h["grad_norm"] for h in history)  # no NaN
