"""Source alignment audit for every REAL / REAL-DERIVED task.

Guardrail G4 (PAIChecker, arXiv:2607.28587): 13.6% of SWE-bench Verified
instances have a PR that does not actually address the linked issue. Citing a URL
is not provenance; it is a claim that has to be checked.

Our exposure differs slightly from PAIChecker's. We do not extract tasks from
issue->PR pairs; we re-instantiate documented bug *patterns* in our own code. So
the failure mode we can suffer is not "the PR fixes something else" but
**"the citation does not say what we claim it says"**. That is the gate audited
here, plus three we can check mechanically.

    A  citation supports the claim   -- manual; each URL was opened and read
    B  pre-fix behaviour reproduces  -- mechanical: buggy fails the hidden suite
    C  gold fix resolves it          -- mechanical: gold passes everything
    D  reduction preserves the root  -- manual, with the mechanism named

Any gate failing => REJECTED_SOURCE. No re-interpretation to save a task.

Usage:  python scripts/source_alignment_audit.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WORK = REPO_ROOT / ".align_work"

# Gate A and D verdicts. Every URL below was fetched and read on 2026-08-10;
# `gate_a_evidence` quotes or paraphrases what the source actually establishes,
# and deliberately records where a citation is weaker than the claim.
MANUAL: dict[str, dict] = {
    # ---------------- Tier E ----------------
    "t1_causal_mask_off_by_one": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/30095",
            "https://github.com/huggingface/transformers/issues/36150",
        ],
        "license": "Apache-2.0 (transformers) - cited as evidence only, no code copied",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "#30095 reports _prepare_4d_attention_mask_for_sdpa being non-causal while used "
            "where causal masking is expected; #36150 reports is_causal=False having no "
            "effect. Both establish that causal-mask construction is a real, recurring "
            "defect surface in production transformer code."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Root cause preserved exactly: an off-by-one in which positions are attendable. "
            "tril(diagonal=1) grants one token of lookahead, the same class of error as the "
            "cited mask-construction defects."
        ),
    },
    "t2_rope_pairing_convention": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/25199",
            "https://github.com/huggingface/transformers/issues/31859",
            "https://github.com/huggingface/transformers/issues/33826",
        ],
        "license": "Apache-2.0 (transformers) - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "#25199 verbatim: 'This is GPT-NeoX style RoPE. But in Meta's official model "
            "implementation, the model adopts GPT-J style RoPE, which processes query and key "
            "vectors in an interleaved way instead of split into two half.' Exactly the "
            "convention mismatch our mutation introduces."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "rotate_half switched from halves pairing to interleaved while the cos/sin cache "
            "still duplicates frequencies for the halves convention - the same cross-convention "
            "mismatch, at tiny scale."
        ),
    },
    "t3_rmsnorm_missing_upcast": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/35945",
            "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py",
        ],
        "license": "Apache-2.0 (transformers) - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "LlamaRMSNorm upstream explicitly casts to float32 before computing the variance "
            "and back afterwards; #35945 concerns that upcast interacting badly with autocast. "
            "The upcast being load-bearing is established by the reference implementation itself."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Removing the upcast reproduces the documented mechanism directly. Measured: fp16 "
            "x**2 exceeds 65504, variance becomes inf, rsqrt(inf)=0, output RMS collapses to "
            "exactly 0.0."
        ),
    },
    "t4_grad_accum_normalization": {
        "sources": [
            "https://huggingface.co/blog/gradient_accumulation",
            "https://unsloth.ai/blog/gradient",
        ],
        "license": "blog posts - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "HuggingFace's own writeup documents the bug and the fix: per-micro-batch mean "
            "normalization is not equivalent to full-batch when token counts differ. Reported "
            "2024-10-15, patched the following day, affected multi-GPU training too. This is "
            "the one task reproducing a specific documented incident verbatim."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Identical mathematics. The invariant is checkable exactly: accumulation over K "
            "micro-batches must equal one backward over the concatenation."
        ),
    },
    "t5_loss_ignore_index_dropped": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/40214",
            "https://huggingface.co/blog/gradient_accumulation",
        ],
        "license": "Apache-2.0 / blog - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "The HF writeup states the denominator is the count of non-padded, non-ignored "
            "tokens - precisely what dropping ignore_index breaks. #40214 documents padding "
            "interacting incorrectly with the objective."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "ignore_index removed from cross_entropy and from the token count, so padded "
            "positions become supervised targets and the loss depends on batch composition."
        ),
    },
    # ---------------- Tier H (redesign round 1) ----------------
    # Each H task inherits its parent's sources; gates A and D are re-stated for
    # the added defect rather than copied, since that defect is what is new.
    "h1_attention_double_defect": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/30095",
            "https://github.com/huggingface/transformers/issues/36150",
            "https://github.com/huggingface/transformers/issues/40214",
        ],
        "license": "Apache-2.0 - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "Inherits m1's mask-construction evidence. The added defect (padding mask "
            "disabled) is covered by #40214, which documents padding interacting "
            "incorrectly with attention."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Two independent real defects in one file: lookahead via tril(diagonal=1) "
            "and padded keys becoming attendable. Both root causes preserved."
        ),
    },
    "h2_position_double_defect": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/25199",
            "https://github.com/huggingface/transformers/issues/31859",
            "https://github.com/huggingface/transformers/issues/33826",
        ],
        "license": "Apache-2.0 - evidence only",
        "gate_a": "PARTIAL",
        "gate_a_evidence": (
            "The pairing-convention half is fully supported (#25199 verbatim). The "
            "added rope_theta 10000 -> 500 change is NOT a documented incident: it is "
            "a plausible porting error we constructed to force a two-file hypothesis. "
            "Recorded as PARTIAL rather than claimed as documented."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Pairing mismatch preserved exactly; the frequency-base change is a second, "
            "independent position-encoding defect in a different file."
        ),
    },
    "h3_accumulation_and_clipping": {
        "sources": [
            "https://huggingface.co/blog/gradient_accumulation",
            "https://unsloth.ai/blog/gradient",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
        ],
        "license": "blog / PyTorch BSD-3 - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "The accumulation denominator defect is the verbatim October-2024 incident "
            "documented by HuggingFace. Clipping after the optimizer step contradicts "
            "the documented contract of clip_grad_norm_, which must precede the update."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Both defects preserved with their real mechanisms; measured signature is "
            "that training still learns, so neither is given away by a single number."
        ),
    },
    "h4_schedule_triple_defect": {
        "sources": [
            "https://github.com/pytorch/pytorch/issues/44511",
            "https://github.com/Lightning-AI/pytorch-lightning/issues/21339",
            "https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/142833",
        ],
        "license": "PyTorch BSD-3 / Lightning Apache-2.0 - evidence only",
        "gate_a": "PARTIAL",
        "gate_a_evidence": (
            "Scheduler mis-stepping is authoritative (PyTorch emits the warning during "
            "this task's run) and weight decay on norm gains is standard practice. The "
            "third defect -- one micro-batch dropped per accumulation window -- is our "
            "construction with no specific cited incident. PARTIAL."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Three independent training-loop defects across two files; measured LR "
            "trace reaches zero at step 20/40 instead of 39/40."
        ),
    },
    "h5_masking_triple_defect": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/40214",
            "https://huggingface.co/blog/gradient_accumulation",
            "https://github.com/huggingface/transformers/issues/30095",
        ],
        "license": "Apache-2.0 / blog - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "Label padding and the loss denominator are covered by the HF writeup and "
            "#40214; dropping the padding mask on the training call is the same defect "
            "class as #30095/#40214, one call site further out."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Three defects spanning data -> model -> train, each preserving its own "
            "root cause; fixing any one leaves the loss batch-composition dependent."
        ),
    },    # ---------------- Tier M ----------------
    "m1_attention_regression": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/30095",
            "https://github.com/huggingface/transformers/issues/36150",
        ],
        "license": "Apache-2.0 - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": "Same sources and reasoning as t1; only the presentation differs.",
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Same root cause as t1, moved into an 8-module package with the location withheld."
        ),
    },
    "m2_position_encoding": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/25199",
            "https://github.com/huggingface/transformers/issues/31859",
            "https://github.com/huggingface/transformers/issues/33826",
        ],
        "license": "Apache-2.0 - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": "Same as t2, verbatim quote confirmed by fetching #25199.",
        "gate_d": "PASS",
        "gate_d_mechanism": "Same mismatch as t2; symptom reduced to a downstream plateau.",
    },
    "m3_gradient_lifecycle": {
        "sources": [
            "https://docs.pytorch.org/docs/stable/optim.html",
            "https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887",
        ],
        "license": "PyTorch docs (BSD-3) / forum - evidence only",
        # Deliberately not PASS. The ordering requirement is documented; this
        # specific incident is not.
        "gate_a": "PARTIAL",
        "gate_a_evidence": (
            "The forum thread confirms the requirement -- 'We're supposed to clear the "
            "gradients each iteration before calling loss.backward() and optimizer.step()' -- "
            "and the PyTorch optim docs specify the zero_grad -> backward -> step order. But "
            "NEITHER source documents this specific failure (zero_grad landing between "
            "backward and step, producing zero-gradient updates). The pattern is real and the "
            "ordering requirement is authoritative; the incident is our construction. "
            "Recorded as PARTIAL rather than upgraded."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "The violated invariant is the documented one. Measured signature: gradient norm "
            "exactly 0.0 at every optimizer step, loss 5.26 -> 5.33 over 40 steps, everything "
            "finite and warning-free."
        ),
    },
    "m4_schedule_accumulation": {
        "sources": [
            "https://github.com/pytorch/pytorch/issues/44511",
            "https://github.com/Lightning-AI/pytorch-lightning/issues/21339",
            "https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/142833",
        ],
        "license": "PyTorch BSD-3 / Lightning Apache-2.0 - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": (
            "PyTorch itself emits 'Detected call of lr_scheduler.step() before "
            "optimizer.step()' and documents that violating the order 'will result in PyTorch "
            "skipping the first value of the learning rate schedule'. That warning fires during "
            "this task's buggy run, so the source is not merely cited, it is triggered."
        ),
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Scheduler over-stepping relative to optimizer steps, plus weight decay applied to "
            "1-D norm gains. Measured: LR reaches 0 at step 20/40 instead of 39/40."
        ),
    },
    "m5_masking_interaction": {
        "sources": [
            "https://github.com/huggingface/transformers/issues/40214",
            "https://huggingface.co/blog/gradient_accumulation",
        ],
        "license": "Apache-2.0 / blog - evidence only",
        "gate_a": "PASS",
        "gate_a_evidence": "Same as t5.",
        "gate_d": "PASS",
        "gate_d_mechanism": (
            "Split across two modules: collate writes pad_token into label positions, and the "
            "objective counts numel(). Same root cause as t5, now requiring two coordinated "
            "edits."
        ),
    },
}


def check_tier_e(task_id: str) -> tuple[str, str, str]:
    """Gates B and C for a Tier E task."""
    from trgym.tasks.build import build_workspace
    from trgym.tasks.registry import get_task
    from trgym.verifier.reward import grade

    spec = get_task(task_id)
    bug = grade(spec, build_workspace(spec, WORK / f"{task_id}_bug", gold=False))
    gold = grade(spec, build_workspace(spec, WORK / f"{task_id}_gold", gold=True))

    gate_b = "PASS" if bug.hardened_reward == 0.0 else "FAIL"
    gate_c = "PASS" if gold.hardened_reward == 1.0 else "FAIL"
    detail = (
        f"buggy hardened={bug.hardened_reward:.0f} naive={bug.naive_reward:.0f}; "
        f"gold hardened={gold.hardened_reward:.0f} naive={gold.naive_reward:.0f}"
    )
    return gate_b, gate_c, detail


def check_tier_m(task_id: str) -> tuple[str, str, str]:
    """Gates B and C for a Tier M task."""
    from trgym.repo.build import build_gold, build_repo
    from trgym.repo.checks import run_repo_checks
    from trgym.tasks.repo_specs import get_repo_task

    spec = get_repo_task(task_id)
    bug_ws = build_repo(spec, WORK / f"{task_id}_bug", gold=False)
    gold_ws = build_gold(spec, WORK / f"{task_id}_gold")

    bug_hidden = run_repo_checks(bug_ws, task_id, spec.hidden_checks)
    bug_visible = run_repo_checks(bug_ws, task_id, spec.visible_checks)
    gold_all = run_repo_checks(gold_ws, task_id, spec.hidden_checks + spec.visible_checks)

    bug_failed = [n for n, ok, _ in bug_hidden if not ok]
    gold_failed = [n for n, ok, _ in gold_all if not ok]

    gate_b = "PASS" if bug_failed else "FAIL"
    gate_c = "PASS" if not gold_failed else "FAIL"
    detail = (
        f"buggy fails hidden={bug_failed}; buggy passes visible="
        f"{all(ok for _, ok, _ in bug_visible)}; gold failures={gold_failed or 'none'}"
    )
    return gate_b, gate_c, detail


def main() -> int:
    from trgym.tasks.registry import TASKS
    from trgym.tasks.repo_specs import REPO_TASKS
    from trgym.tasks.repo_specs_h import REPO_TASKS_H

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    tiers = {t.task_id: ("E", t) for t in TASKS}
    tiers.update({t.task_id: ("M", t) for t in REPO_TASKS})
    tiers.update({t.task_id: ("H", t) for t in REPO_TASKS_H})

    rows = []
    for task_id, (tier, spec) in tiers.items():
        manual = MANUAL.get(task_id)
        if manual is None:
            rows.append(
                {"task_id": task_id, "tier": tier, "verdict": "REJECTED_SOURCE",
                 "reviewer_notes": "no manual audit entry"}
            )
            continue

        gate_b, gate_c, detail = (
            check_tier_e(task_id) if tier == "E" else check_tier_m(task_id)
        )
        gates = {
            "A_citation_supports_claim": manual["gate_a"],
            "B_prefix_reproduces": gate_b,
            "C_gold_fix_resolves": gate_c,
            "D_reduction_preserves_root_cause": manual["gate_d"],
        }
        failed = [k for k, v in gates.items() if v == "FAIL"]
        partial = [k for k, v in gates.items() if v == "PARTIAL"]

        if failed:
            verdict = "REJECTED_SOURCE"
        elif partial:
            verdict = "ACCEPTED_WITH_CAVEAT"
        else:
            verdict = "ACCEPTED"

        rows.append(
            {
                "task_id": task_id,
                "tier": tier,
                "family_id": spec.family_id,
                "declared_provenance": spec.provenance,
                "sources": " | ".join(manual["sources"]),
                "license": manual["license"],
                **gates,
                "verdict": verdict,
                "mechanical_detail": detail,
                "gate_a_evidence": manual["gate_a_evidence"],
                "gate_d_mechanism": manual["gate_d_mechanism"],
            }
        )
        print(
            f"{verdict:22s} {tier}  {task_id:30s} "
            f"A={gates['A_citation_supports_claim']:7s} B={gate_b:4s} C={gate_c:4s} "
            f"D={gates['D_reduction_preserves_root_cause']}"
        )

    accepted = [r for r in rows if r["verdict"] == "ACCEPTED"]
    caveat = [r for r in rows if r["verdict"] == "ACCEPTED_WITH_CAVEAT"]
    rejected = [r for r in rows if r["verdict"] == "REJECTED_SOURCE"]

    summary = {
        "n_tasks": len(rows),
        "accepted": len(accepted),
        "accepted_with_caveat": len(caveat),
        "rejected_source": len(rejected),
        "caveat_task_ids": [r["task_id"] for r in caveat],
        "rejected_task_ids": [r["task_id"] for r in rejected],
        "gate_a_pass": sum(1 for r in rows if r.get("A_citation_supports_claim") == "PASS"),
        "gate_a_partial": sum(
            1 for r in rows if r.get("A_citation_supports_claim") == "PARTIAL"
        ),
    }
    print("\n" + "=" * 70)
    for k, v in summary.items():
        print(f"{k:28s} {v}")

    fields = [
        "task_id", "tier", "family_id", "declared_provenance", "sources", "license",
        "A_citation_supports_claim", "B_prefix_reproduces", "C_gold_fix_resolves",
        "D_reduction_preserves_root_cause", "verdict", "mechanical_detail",
        "gate_a_evidence", "gate_d_mechanism", "reviewer_notes",
    ]
    out_csv = REPO_ROOT / "SOURCE_ALIGNMENT_AUDIT.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (REPO_ROOT / "artifacts" / "source_alignment_audit.json").write_text(
        json.dumps({"summary": summary, "tasks": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_csv}")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
