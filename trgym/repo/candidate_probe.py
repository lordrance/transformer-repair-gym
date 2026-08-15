"""Runs INSIDE the candidate container. Computes observations; decides nothing.

This file is mounted into the sandbox on its own, at a path that contains no other part
of this repository. It must therefore never import `trgym`: by the time it runs, the
oracle, the hidden checks, the gold tree and the pristine template are all absent from
the filesystem it can see. That absence is the boundary. Everything here is duplicated
from the trusted side on purpose -- `RepoModules`, `_seeded`, the fixture shapes -- and
the duplication is the price of the candidate never holding a handle to the grader.

The contract is one-directional and narrow:

    stdin   {"nonce", "workspace", "inputs": {...}}   public task inputs only
    stdout  <<<TRGYM_OBS:{nonce}>>>{"observations", "errors"}

No entry in `observations` is a verdict. Every predicate is applied by the trusted
comparator after this process has exited, because a verdict computed here is a verdict
the candidate's own module-level code could have written.

A note on what this can and cannot buy. Observations whose ground truth is *gold* cannot
be faked: the candidate does not have gold, so it cannot know what to claim. Observations
that are purely internal to the candidate (its own gradient norms, its own return types)
are forgeable by a candidate that fabricates them -- moving the predicate out of this
process does not change that, and SECURITY_MODEL.md says so plainly rather than implying
the split fixes it.
"""

from __future__ import annotations

import base64
import importlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import torch

VISIBLE_SHAPE = (2, 12)
HIDDEN_SHAPES = ((1, 9), (2, 17), (3, 5), (1, 31), (4, 12))

_COUNTER = itertools.count()


# --------------------------------------------------------------------------- #
# Wire format. Mirrors trgym/repo/obs_protocol.py, which validates what we emit.
# --------------------------------------------------------------------------- #
_DTYPES = {"torch.float32": "float32", "torch.int64": "int64", "torch.bool": "bool"}


def enc(value):
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        name = _DTYPES.get(str(t.dtype))
        if name is None:
            t = t.to(torch.float32)
            name = "float32"
        return {
            "__tensor__": {
                "dtype": name,
                "shape": list(t.shape),
                "data": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
            }
        }
    if isinstance(value, (list, tuple)):
        return [enc(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def dec(value):
    """Decode a trusted-supplied input. Trusted output is not attacker data."""
    if isinstance(value, dict) and "__tensor__" in value:
        spec = value["__tensor__"]
        raw = base64.b64decode(spec["data"])
        if not raw:
            return torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"]))
        flat = torch.frombuffer(bytearray(raw), dtype=getattr(torch, spec["dtype"]))
        return flat.reshape(spec["shape"]).clone()
    if isinstance(value, list):
        return [dec(v) for v in value]
    return value


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class RepoModules:
    """Import `tinygpt` from a directory under a unique alias.

    A copy of the trusted-side class. It cannot be imported from there: `trgym` is not
    on this container's filesystem, which is the property this whole design exists for.
    """

    def __init__(self, root) -> None:
        self.root = Path(root).resolve()
        self.alias = f"_probe_repo_{next(_COUNTER)}"
        self._saved_path = None

    def __enter__(self) -> "RepoModules":
        self._saved_path = list(sys.path)
        sys.path.insert(0, str(self.root))
        for name in [n for n in sys.modules if n == "tinygpt" or n.startswith("tinygpt.")]:
            del sys.modules[name]
        # Delete cached bytecode before importing. `invalidate_caches()` resets finder
        # caches; whether a .pyc is reused is decided by the source (mtime, size) pair in
        # its header. A candidate edit that keeps the file the same size and lands in the
        # same mtime second would silently run the OLD code -- grading a correct repair as
        # broken. Mirrors trgym/repo/visible_runtime.py::_purge_bytecode.
        for cache in self.root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        importlib.invalidate_caches()
        self.pkg = importlib.import_module("tinygpt")
        self.config = importlib.import_module("tinygpt.config")
        self.norm = importlib.import_module("tinygpt.norm")
        self.positional = importlib.import_module("tinygpt.positional")
        self.attention = importlib.import_module("tinygpt.attention")
        self.model = importlib.import_module("tinygpt.model")
        self.data = importlib.import_module("tinygpt.data")
        self.optim = importlib.import_module("tinygpt.optim")
        self.train = importlib.import_module("tinygpt.train")
        return self

    def __exit__(self, *_exc) -> None:
        for name in [n for n in sys.modules if n == "tinygpt" or n.startswith("tinygpt.")]:
            del sys.modules[name]
        if self._saved_path is not None:
            sys.path[:] = self._saved_path


# --------------------------------------------------------------------------- #
# Observation groups
# --------------------------------------------------------------------------- #
GROUPS: dict = {}


def group(name: str):
    def deco(fn):
        GROUPS[name] = fn
        return fn

    return deco


@group("gold_logits")
def obs_gold_logits(ws, inputs):
    """Weights and logits. The trusted side loads these into the gold model and compares.

    Note the direction: the candidate's `state_dict` travels outward. The original check
    loaded gold's weights into the candidate; doing it the other way round tests exactly
    the same thing -- two implementations on one set of weights -- without gold moving in.
    """
    ids_list = [dec(t) for t in inputs["gold_logit_ids"]]
    with RepoModules(ws) as cand:
        torch.manual_seed(20260810)
        model = cand.model.TinyGPT(cand.config.Config())
        model.eval()
        sd = model.state_dict()
        out = {
            "sd_names": list(sd.keys()),
            "sd_values": [sd[k] for k in sd],
            "logits": [],
        }
        for ids in ids_list:
            with torch.no_grad():
                out["logits"].append(model(ids))
        return out


@group("strict_causality")
def obs_strict_causality(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(1)
        model = cand.model.TinyGPT(cfg).eval()
        out = {"lengths": [], "base": [], "positions": [], "bumped": []}
        for s in (5, 13, 24):
            ids = torch.randint(1, cfg.vocab_size, (1, s), generator=_seeded(100 + s))
            with torch.no_grad():
                base = model(ids)
            positions = sorted({1, s // 2, s - 1})
            bumped_out = []
            for pos in positions:
                bumped = ids.clone()
                bumped[0, pos] = (bumped[0, pos] % (cfg.vocab_size - 1)) + 1
                with torch.no_grad():
                    bumped_out.append(model(bumped))
            out["lengths"].append(s)
            out["base"].append(base)
            out["positions"].append(positions)
            out["bumped"].append(bumped_out)
        return out


@group("padding_keys_masked")
def obs_padding_keys_masked(ws, inputs):
    # The fixture is supplied by the trusted side (v0.2-B). Generating it here would make
    # gold's answer uncomputable out there, which is what left this check forgeable.
    q, k = dec(inputs["mask_q"]), dec(inputs["mask_k"])
    v, v2, pad = dec(inputs["mask_v"]), dec(inputs["mask_v2"]), dec(inputs["mask_pad"])
    with RepoModules(ws) as cand:
        return {
            "out1": cand.attention.causal_attention(q, k, v, pad),
            "out2": cand.attention.causal_attention(q, k, v2, pad),
        }


@group("rope_relative")
def obs_rope_relative(ws, inputs):
    q, k = dec(inputs["rope_q"]), dec(inputs["rope_k"])
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(40, 16, 10000.0)

        def dot(m, n):
            qm, _ = cand.positional.apply_rope(q, q, cos[m : m + 1], sin[m : m + 1])
            kn, _ = cand.positional.apply_rope(k, k, cos[n : n + 1], sin[n : n + 1])
            return float((qm * kn).sum())

        pairs = (((9, 4), (14, 9)), ((20, 3), (33, 16)))
        return {
            "pairs": [[list(a), list(b)] for a, b in pairs],
            "dots": [[dot(*a), dot(*b)] for a, b in pairs],
        }


@group("rope_norm")
def obs_rope_norm(ws, inputs):
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(16, 16, 10000.0)
        q = torch.randn(2, 3, 16, 16, generator=_seeded(17))
        q_rot, _ = cand.positional.apply_rope(q, q, cos, sin)
        # `q_rot` itself is returned, not only its norm: the trusted side compares the
        # rotated tensor against gold's, which a norm alone cannot support (v0.2-B).
        return {"q_norm": q.norm(dim=-1), "q_rot_norm": q_rot.norm(dim=-1), "q_rot": q_rot}


@group("supervised_token_count")
def obs_supervised_token_count(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        seeds, n_sup, n_tok = [], [], []
        for seed in (7, 21):
            batch = cand.data.make_batches(cfg, 1, seed=seed)[0]
            torch.manual_seed(5)
            model = cand.model.TinyGPT(cfg)
            _, tokens = model.loss_sum(batch.input_ids, batch.labels, batch.padding_mask)
            seeds.append(seed)
            n_sup.append(int(batch.n_supervised))
            n_tok.append(int(tokens))
        return {"seeds": seeds, "n_supervised": n_sup, "loss_n_tokens": n_tok}


@group("padding_does_not_change_loss")
def obs_padding_does_not_change_loss(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(5)
        model = cand.model.TinyGPT(cfg)
        g = _seeded(41)
        core = [cand.data.make_sequence(cfg, 11, g) for _ in range(2)]

        batch_a = cand.data.collate(cfg, core)
        loss_a, n_a = model.loss_sum(
            batch_a.input_ids, batch_a.labels, batch_a.padding_mask
        )
        padded = core + [cand.data.make_sequence(cfg, 20, g)]
        batch_b = cand.data.collate(cfg, padded)
        loss_b, _ = model.loss_sum(
            batch_b.input_ids[:2], batch_b.labels[:2], batch_b.padding_mask[:2]
        )
        n_b = int((batch_b.labels[:2, 1:] != cfg.ignore_index).sum())
        return {
            "n_a": int(n_a),
            "n_b": n_b,
            "loss_a": float(loss_a.detach()),
            "loss_b": float(loss_b.detach()),
        }


@group("no_pad_probability_mass")
def obs_no_pad_probability_mass(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        optimizer = cand.optim.make_optimizer(model, cfg)
        for batch in cand.data.make_batches(cfg, 25, seed=cfg.seed + 1):
            loss_sum, n_tokens = model.loss_sum(
                batch.input_ids, batch.labels, batch.padding_mask
            )
            if int(n_tokens) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            (loss_sum / int(n_tokens)).backward()
            optimizer.step()
        probe = cand.data.collate(cfg, [cand.data.make_sequence(cfg, 12, _seeded(3))])
        with torch.no_grad():
            probs = torch.softmax(model(probe.input_ids)[0, -1], dim=-1)
        return {"pad_mass": float(probs[cfg.pad_token])}


@group("training_history")
def obs_training_history(ws, inputs):
    """Serves both the finiteness check and the convergence check."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        history = cand.train.train(cfg, steps=cfg.total_steps, verbose=False)
        return {
            "loss": [float(v) for v in history["loss"]],
            "grad_norm": [float(v) for v in history["grad_norm"]],
            "final_loss": float(history["final_loss"]),
            "total_steps": int(cfg.total_steps),
            "vocab_size": int(cfg.vocab_size),
        }


@group("lr_schedule")
def obs_lr_schedule(ws, inputs):
    with RepoModules(ws) as cand:
        trace = cand.train.train(cand.config.Config(), steps=20, verbose=False)["lr"]
        return {"lr": [float(v) for v in trace], "steps": 20}


@group("weight_decay_groups")
def obs_weight_decay_groups(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        optimizer = cand.optim.make_optimizer(model, cfg)
        gains = {id(p) for _, p in model.named_parameters() if p.ndim < 2}
        decays, leaked = [], []
        for grp in optimizer.param_groups:
            decays.append(float(grp.get("weight_decay", 0.0)))
            leaked.append(sum(1 for p in grp["params"] if id(p) in gains))
        return {"weight_decay": decays, "n_gains_in_group": leaked}


@group("grad_norms_at_step_time")
def obs_grad_norms_at_step_time(ws, inputs):
    """Serves the clipping check and the gradients-reach-optimizer check."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        import tinygpt.train as train_mod

        observed: list[float] = []
        original_step = torch.optim.AdamW.step

        def spy(self, *args, **kwargs):
            total = 0.0
            for grp in self.param_groups:
                for p in grp["params"]:
                    if p.grad is not None:
                        total += float(p.grad.detach().norm() ** 2)
            observed.append(math.sqrt(total))
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = spy
        try:
            train_mod.train(cfg, steps=8, verbose=False)
        finally:
            torch.optim.AdamW.step = original_step
        return {"observed": observed, "grad_clip": float(cfg.grad_clip)}


@group("grad_accum")
def obs_grad_accum(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 3, seed=cfg.seed + 7)

        def flat_grads():
            return torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                    for p in model.parameters()
                ]
            )

        cand.train.accumulate_gradients(model, batches)
        g_accum = flat_grads()

        model.zero_grad(set_to_none=True)
        total_sum, total_tokens = None, 0
        for b in batches:
            loss_sum, n_tok = model.loss_sum(b.input_ids, b.labels, b.padding_mask)
            total_sum = loss_sum if total_sum is None else total_sum + loss_sum
            total_tokens += int(n_tok)
        if total_tokens == 0:
            return {"total_tokens": 0, "g_accum": g_accum, "g_full": g_accum}
        (total_sum / total_tokens).backward()
        return {"total_tokens": total_tokens, "g_accum": g_accum, "g_full": flat_grads()}


@group("padding_mask_reaches_model")
def obs_padding_mask_reaches_model(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batch = cand.data.make_batches(cfg, 1, seed=cfg.seed + 11)[0]
        has_padding = not bool(batch.padding_mask.all())

        with torch.no_grad():
            masked = model(batch.input_ids, batch.padding_mask)
            unmasked = model(batch.input_ids, None)

        seen = {}
        original = cand.attention.causal_attention

        def spy(*args, **kwargs):
            seen["padding_mask"] = kwargs.get(
                "padding_mask", args[3] if len(args) > 3 else None
            )
            return original(*args, **kwargs)

        cand.attention.causal_attention = spy
        try:
            cand.train.train(cfg, steps=1, verbose=False)
        finally:
            cand.attention.causal_attention = original
        return {
            "has_padding": has_padding,
            "masked": masked,
            "unmasked": unmasked,
            "mask_reached_attention": seen.get("padding_mask") is not None,
        }


@group("contract_types")
def obs_contract_types(ws, inputs):
    """Type names as strings. The trusted side owns what the contract requires."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 2, seed=cfg.seed + 5)
        b = batches[0]

        out: dict = {
            "input_ids_type": type(b.input_ids).__name__,
            "labels_type": type(b.labels).__name__,
            "padding_mask_type": type(b.padding_mask).__name__,
            "n_supervised_type": type(b.n_supervised).__name__,
        }
        logits = model(b.input_ids, b.padding_mask)
        out["logits_type"] = type(logits).__name__
        out["logits_dtype"] = str(logits.dtype)
        out["logits_shape"] = [int(v) for v in logits.shape]
        out["want_logits_shape"] = [
            int(b.input_ids.shape[0]), int(b.input_ids.shape[1]), int(cfg.vocab_size)
        ]

        res = model.loss_sum(b.input_ids, b.labels, b.padding_mask)
        out["loss_sum_is_tuple"] = isinstance(res, tuple)
        out["loss_sum_len"] = len(res) if isinstance(res, tuple) else -1
        if isinstance(res, tuple) and len(res) == 2:
            loss_val, n_tok = res
            out["loss_value_type"] = type(loss_val).__name__
            out["n_tokens_type"] = type(n_tok).__name__
            try:
                out["n_tokens_value"] = int(n_tok)
            except Exception:
                out["n_tokens_value"] = -1

        opt = cand.optim.make_optimizer(model, cfg)
        out["optimizer_is_torch_optimizer"] = isinstance(opt, torch.optim.Optimizer)
        out["optimizer_type"] = type(opt).__name__
        sched = cand.optim.make_scheduler(opt, cfg)
        out["scheduler_missing"] = [
            a for a in ("step", "get_last_lr") if not callable(getattr(sched, a, None))
        ]

        loss = cand.train.accumulate_gradients(model, batches)
        out["accum_exact_type"] = type(loss).__name__

        history = cand.train.train(cfg, steps=3, verbose=False)
        out["history_type"] = type(history).__name__
        if isinstance(history, dict):
            out["final_loss_exact_type"] = type(history.get("final_loss")).__name__
            for key in ("loss", "lr", "grad_norm"):
                seq = history.get(key)
                out[f"{key}_is_list"] = isinstance(seq, list)
                out[f"{key}_all_float"] = bool(
                    isinstance(seq, list) and all(type(v) is float for v in seq)
                )
            try:
                json.dumps({k: v for k, v in history.items() if k != "all_finite"})
                out["history_json_serialisable"] = True
                out["history_json_error"] = ""
            except TypeError as exc:
                out["history_json_serialisable"] = False
                out["history_json_error"] = str(exc)[:300]
        return out


# ------------------------------------------------------------------ visible suite
@group("visible_smoke")
def obs_visible_smoke(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        b, s = VISIBLE_SHAPE
        ids = torch.randint(1, cfg.vocab_size, (b, s), generator=_seeded(1234))
        out = model(ids)
        return {
            "shape": [int(v) for v in out.shape],
            "want_shape": [b, s, int(cfg.vocab_size)],
            "all_finite": bool(torch.isfinite(out).all()),
        }


@group("visible_loss_is_finite")
def obs_visible_loss_is_finite(ws, inputs):
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        batch = cand.data.make_batches(cfg, 1, seed=1234)[0]
        loss_sum, n_tokens = model.loss_sum(
            batch.input_ids, batch.labels, batch.padding_mask
        )
        return {
            "loss_finite": bool(torch.isfinite(loss_sum)),
            "n_tokens": int(n_tokens),
        }


@group("visible_single_token_attention")
def obs_visible_single_token_attention(ws, inputs):
    with RepoModules(ws) as cand:
        q = torch.randn(1, 1, 1, 16, generator=_seeded(11))
        return {"q": q, "out": cand.attention.causal_attention(q, q, q)}


@group("visible_rope_position_zero")
def obs_visible_rope_position_zero(ws, inputs):
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(4, 16, 10000.0)
        return {"cos0": cos[0], "sin0": sin[0]}


@group("visible_short_train_runs")
def obs_visible_short_train_runs(ws, inputs):
    with RepoModules(ws) as cand:
        history = cand.train.train(cand.config.Config(), steps=5, verbose=False)
        return {"loss": [float(v) for v in history["loss"]]}


# --------------------------------------------------------------------------- #
def main() -> int:
    req = json.loads(sys.stdin.read())
    nonce = req["nonce"]
    ws = req["workspace"]
    inputs = req.get("inputs") or {}
    wanted = req.get("groups") or list(GROUPS)

    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    observations: dict = {}
    errors: dict = {}
    for name in wanted:
        fn = GROUPS.get(name)
        if fn is None:
            errors[name] = f"unknown observation group {name!r}"
            continue
        try:
            for key, value in fn(ws, inputs).items():
                observations[f"{name}.{key}"] = enc(value)
        except Exception as exc:  # noqa: BLE001 - candidate code is arbitrary
            errors[name] = f"{type(exc).__name__}: {exc}"[:2000]

    sys.stdout.write(
        "<<<TRGYM_OBS:" + nonce + ">>>"
        + json.dumps({"observations": observations, "errors": errors})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
