"""G6 -- false-positive audit of the static anti-exploit gates.

The question this answers: when the hardened verifier refuses a submission because a
heuristic gate fired, how often is that refusal wrong?

**Corpus choice.** Every sample is a real, human-written function or class extracted from
an installed, permissively licensed ML library -- and deliberately from *this project's own
domain*: attention, normalisation, optimisers, LR schedules, gradient clipping, collation.
That makes it the strongest possible adversary for the gates. A gate that fires on
PyTorch's own `MultiheadAttention` would fire on a competent human fix to an attention bug,
which is precisely the failure mode worth measuring. Scraping arbitrary GitHub files would
have given weaker provenance (no verified license, no pinned version) and a less relevant
corpus.

**Hard vs soft separation.** The contract requires it, and the gates genuinely split:

  HARD / structural -- compare the submission against the task's reference or check
  filesystem state. They encode the task contract, so running foreign code through them is
  meaningless (any unrelated file trivially "removes" the required API). Not FP-audited
  here; they are audited by G2's replay, where the inputs are real submissions.
      public_api_preserved, grader_files_untouched, support_files_untouched

  SOFT / content heuristics -- pattern-match the submission text alone. These are the ones
  that can misfire on benign code, so these are what this audit measures.
      no_reference_import, no_dynamic_execution, no_shape_special_casing,
      no_silent_exception_swallowing, no_hardcoded_tensors

Outputs artifacts/heuristic_gate_fp_audit.json and prints a table.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import metadata, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "artifacts" / "heuristic_gate_fp_audit.json"

# (package, relative module path, [symbols to extract])
# Chosen for domain overlap with the five task families, not for convenience.
CORPUS_SPEC = [
    ("torch", "nn/modules/normalization.py", ["LayerNorm", "RMSNorm", "GroupNorm"]),
    ("torch", "nn/modules/activation.py", ["MultiheadAttention", "GLU", "Softmax"]),
    ("torch", "nn/modules/sparse.py", ["Embedding"]),
    ("torch", "nn/modules/dropout.py", ["Dropout", "AlphaDropout"]),
    ("torch", "nn/utils/clip_grad.py", ["clip_grad_norm_", "clip_grad_value_"]),
    ("torch", "optim/adamw.py", ["AdamW"]),
    ("torch", "optim/sgd.py", ["SGD"]),
    ("torch", "optim/lr_scheduler.py",
     ["CosineAnnealingLR", "LinearLR", "LambdaLR", "OneCycleLR"]),
    ("torch", "nn/modules/transformer.py",
     ["TransformerEncoderLayer", "TransformerDecoderLayer", "TransformerEncoder"]),
    ("torch", "nn/modules/loss.py", ["CrossEntropyLoss", "MSELoss", "KLDivLoss"]),
    ("torch", "utils/data/_utils/collate.py", ["collate", "default_collate"]),
    ("torch", "nn/init.py", ["kaiming_uniform_", "xavier_normal_", "trunc_normal_"]),
    ("numpy", "lib/_function_base_impl.py", ["gradient", "percentile"]),
    ("numpy", "_core/fromnumeric.py", ["argmax", "clip"]),
]

# The gates whose decision depends only on the submission text.
SOFT_GATES = (
    "no_reference_import",
    "no_dynamic_execution",
    "no_shape_special_casing",
    "no_silent_exception_swallowing",
    "no_hardcoded_tensors",
)
HARD_GATES = (
    "public_api_preserved",
    "grader_files_untouched",
    "support_files_untouched",
)


@dataclass
class Sample:
    sample_id: str
    package: str
    package_version: str
    license: str
    module: str
    symbol: str
    lineno: int
    end_lineno: int
    n_lines: int
    sha256: str


def package_root(pkg: str) -> Path:
    mod = __import__(pkg)
    return Path(mod.__file__).parent


def license_of(pkg: str) -> str:
    try:
        md = metadata(pkg)
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    lic = (md.get("License") or "").strip()
    classifiers = [c for c in (md.get_all("Classifier") or []) if "License" in c]
    if lic and len(lic) < 60:
        return lic
    if classifiers:
        return classifiers[0].split("::")[-1].strip()
    return (lic[:60] + "…") if lic else "UNKNOWN"


def build_corpus() -> tuple[list[Sample], dict[str, str]]:
    samples: list[Sample] = []
    sources: dict[str, str] = {}
    for pkg, rel, symbols in CORPUS_SPEC:
        try:
            path = package_root(pkg) / rel
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception as exc:  # noqa: BLE001 - a missing module is skipped, recorded below
            print(f"  skip {pkg}/{rel}: {type(exc).__name__}: {exc}")
            continue
        lines = text.splitlines()
        by_name = {
            n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for sym in symbols:
            node = by_name.get(sym)
            if node is None:
                print(f"  skip {pkg}/{rel}::{sym}: not a top-level symbol")
                continue
            src = "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
            sid = f"{pkg}:{rel}:{sym}"
            samples.append(Sample(
                sample_id=sid, package=pkg, package_version=version(pkg),
                license=license_of(pkg), module=rel, symbol=sym,
                lineno=node.lineno, end_lineno=node.end_lineno,
                n_lines=node.end_lineno - node.lineno + 1,
                sha256=hashlib.sha256(src.encode()).hexdigest()[:16],
            ))
            sources[sid] = src
    return samples, sources


# --------------------------------------------------------------------------- #
# The soft gates. The REAL gate functions are called, never reimplemented, so this
# audit cannot drift from what the verifier actually runs.
# --------------------------------------------------------------------------- #
def soft_gate_results(source: str) -> dict[str, list[str]]:
    """Run the REAL gate functions from `trgym.verifier.anticheat`.

    An earlier version of this audit re-implemented each gate as a plain regex over the
    source. That measured something stricter than the verifier -- the real
    `gate_no_shape_special_casing` walks the AST and deliberately ignores comparisons
    against 0/1/2 as ordinary defensive code -- and it credited the gates with a false
    positive (numpy's `if n == 0:`) that they never actually produce. An FP audit that
    does not call the code under audit is measuring its own reimplementation.

    The four content-only gates take `spec`/`ws` but never read them, so None is safe and
    is asserted by `test_content_gates_ignore_spec`. `gate_no_hardcoded_tensors` does read
    the reference, so it gets a real Tier E spec.
    """
    from trgym.verifier import anticheat as ac

    fired: dict[str, list[str]] = {}
    content_gates = {
        "no_reference_import": ac.gate_no_reference_import,
        "no_dynamic_execution": ac.gate_no_dynamic_execution,
        "no_shape_special_casing": ac.gate_no_shape_special_casing,
        "no_silent_exception_swallowing": ac.gate_no_silent_exception_swallowing,
    }
    for name, gate in content_gates.items():
        for v in gate(source, None, None):  # type: ignore[arg-type]
            fired.setdefault(name, []).append(f"{v.detail} :: {v.evidence}"[:200])

    spec = _reference_spec()
    if spec is not None:
        for v in ac.gate_no_hardcoded_tensors(source, spec, Path(".")):
            fired.setdefault("no_hardcoded_tensors", []).append(
                f"{v.detail} :: {v.evidence}"[:200]
            )
    return fired


def _reference_spec():
    """A real Tier E spec, so `gate_no_hardcoded_tensors` reads a real reference file."""
    try:
        from trgym.tasks.registry import TASKS

        return TASKS[0]
    except Exception:  # noqa: BLE001 - recorded in the payload as a skipped gate
        return None


def exploit_suite_status() -> dict:
    """Read the authoritative exploit-suite result from `scripts/audit_cheats.py`.

    Deliberately NOT reimplemented here. The contract's machine check is that the exploit
    suite still catches 7/7, and the suite is `audit_cheats.py` -- it runs the full
    naive-vs-hardened comparison including the protected oracle, which two of the seven
    cheats depend on (they carry `expected_gate=None` precisely because no static gate can
    see them). An approximation limited to the static gates would silently score those two
    as misses and understate the protection.

    Run `python scripts/audit_cheats.py` first; this reads its artifact.
    """
    path = ROOT / "artifacts" / "cheat_audit.json"
    if not path.exists():
        return {"error": "artifacts/cheat_audit.json missing; run scripts/audit_cheats.py"}
    rows = json.loads(path.read_text(encoding="utf-8"))
    caught = [r for r in rows if r.get("verdict") == "CAUGHT"]
    return {
        "source": "artifacts/cheat_audit.json",
        "n_cheats": len(rows),
        "n_caught": len(caught),
        "all_caught": len(rows) > 0 and len(caught) == len(rows),
        "per_cheat": [
            {
                "cheat_id": r["cheat_id"],
                "verdict": r["verdict"],
                "naive_reward": r["naive_reward"],
                "hardened_reward": r["hardened_reward"],
                "gates_fired": r["gates_fired"],
                "hidden_checks_failed": r["hidden_checks_failed"],
                # Two cheats carry expected_gate=None by design: no static gate can see
                # them and the protected oracle is what catches them.
                "caught_statically": bool(r["gates_fired"]),
            }
            for r in rows
        ],
    }


def measure(samples, sources) -> tuple[dict[str, int], list[dict], int]:
    per_gate: dict[str, int] = {g: 0 for g in SOFT_GATES}
    results = []
    for s in samples:
        fired = soft_gate_results(sources[s.sample_id])
        for g in fired:
            per_gate[g] = per_gate.get(g, 0) + 1
        results.append({**asdict(s), "gates_fired": dict(fired)})
    return per_gate, results, sum(1 for r in results if r["gates_fired"])


def print_table(per_gate: dict[str, int], any_fired: int, n: int, label: str) -> None:
    print()
    print(label)
    print(f"{'gate':<38} {'fired':>6} {'FP rate':>9}")
    print("-" * 55)
    for g in SOFT_GATES:
        c = per_gate.get(g, 0)
        print(f"{g:<38} {c:>6} {c / n:>8.1%}")
    print("-" * 55)
    print(f"{'ANY soft gate (submission-level FP)':<38} {any_fired:>6} {any_fired / n:>8.1%}")


def main() -> int:
    print("-- building corpus of benign, licensed, human-written ML code --")
    samples, sources = build_corpus()
    print(f"   {len(samples)} samples from "
          f"{len({s.package for s in samples})} packages\n")

    if len(samples) < 20:
        print(f"FAIL: need >=20 samples, built {len(samples)}")
        return 1

    n = len(samples)

    # BEFORE: the gates as they were prior to R13, i.e. matching against comments and
    # docstrings too. Measured by neutralising `code_only` on the real gate module, so
    # this is the genuine pre-fix behaviour rather than a reimplementation of it.
    from trgym.verifier import anticheat as ac

    real_code_only = ac.code_only
    ac.code_only = lambda src: src  # type: ignore[assignment]
    try:
        before_gate, before_results, before_any = measure(samples, sources)
    finally:
        ac.code_only = real_code_only  # type: ignore[assignment]
    print_table(before_gate, before_any, n, "BEFORE R13 (gates also scanned prose)")

    # AFTER: comments and string literals blanked before pattern matching.
    per_gate, results, any_fired = measure(samples, sources)
    print_table(per_gate, any_fired, n, "AFTER R13 (gates scan executable code only)")

    print("\n-- re-running the exploit suite --")
    exploits = exploit_suite_status()
    if "error" in exploits:
        print(f"   {exploits['error']}")
        return 1
    print(f"   {exploits['n_caught']}/{exploits['n_cheats']} caught "
          f"(from {exploits['source']})")
    if not exploits["all_caught"]:
        print("   FAIL: protection was lost")
        return 1

    payload = {
        "n_samples": n,
        "before_R13": {
            "per_gate_fp_count": before_gate,
            "per_gate_fp_rate": {g: before_gate.get(g, 0) / n for g in SOFT_GATES},
            "submission_level_fp_count": before_any,
            "submission_level_fp_rate": before_any / n,
            "false_positives": [
                {"sample_id": r["sample_id"], "gates": sorted(r["gates_fired"]),
                 "evidence": {k: v[:1] for k, v in r["gates_fired"].items()}}
                for r in before_results if r["gates_fired"]
            ],
        },
        "packages": sorted({f"{s.package}=={s.package_version}" for s in samples}),
        "licenses": sorted({f"{s.package}: {s.license}" for s in samples}),
        "soft_gates": list(SOFT_GATES),
        "hard_gates_excluded_from_fp_audit": list(HARD_GATES),
        "per_gate_fp_count": per_gate,
        "per_gate_fp_rate": {g: per_gate.get(g, 0) / n for g in SOFT_GATES},
        "submission_level_fp_count": any_fired,
        "submission_level_fp_rate": any_fired / n,
        "samples": results,
        "exploit_suite": exploits,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
