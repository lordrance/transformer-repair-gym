# LOCALIZATION_SCALE_REPORT

G4. Tier S asks a question Tier M cannot: when the defect is one line in a 48-file package
and the agent cannot read the whole thing, can it still find it?

Everything below is generated from `artifacts/tier_s_primary.jsonl` via
`scripts/tier_s_report.py`, with the task definitions frozen in `artifacts/tier_s_spec.json`
**before** the first paid call.

---

## 1. The tasks

Three tasks, one mutated file each, built from `trgym/repo_template_s`. The package keeps
the same eight public modules as Tier M — `config`, `norm`, `positional`, `attention`,
`model`, `data`, `optim`, `train` — but they are thin facades over nine subpackages
holding the implementation. `tinygpt/attention.py` delegates to `_layers/attention.py`,
which calls `_ops/masking.py`. The symptom points at attention; the defect is four levels
down.

Reusing the public API this way is what makes gold-PASS/no-op-FAIL *inherited* rather than
re-derived: the same hidden checks that grade Tier M grade Tier S unchanged.

| task | defect | file |
|---|---|---|
| `s1_causal_mask_offbyone` | `tril(diagonal=0)` → `tril(diagonal=1)` | `tinygpt/_ops/masking.py` |
| `s2_padding_supervised_as_labels` | padded labels get `pad_token`, not `ignore_index` | `tinygpt/_data/collate.py` |
| `s3_warmup_offbyone` | `(step+1)/warmup` → `step/warmup` | `tinygpt/_optim/schedule.py` |

### Freeze preflight — measured, not asserted

`scripts/tier_s_freeze.py`, run before any model call:

| | s1 | s2 | s3 |
|---|---|---|---|
| Python files | 48 | 48 | 48 |
| import edges | 89 | 89 | 89 |
| orphan (unreferenced) files | 0 | 0 | 0 |
| relevant files | 1 | 1 | 1 |
| only relevant files differ from gold | yes | yes | yes |
| gold PASSES hidden suite | yes | yes | yes |
| no-op FAILS hidden suite | yes | yes | yes |
| which checks the defect trips | `matches_gold_logits`, `strict_causality` | `padding_does_not_change_loss`, `supervised_token_count` | `lr_schedule_matches_gold` |

The anti-padding requirement is checked as a **graph traversal**, not a grep: every module
must be reachable from the package root or one of the eight facades. `import x` in a file
nothing imports would not count.

That check earned its keep. Its first run reported `tinygpt/_util/seeding.py` as an orphan,
which was two bugs at once — the resolver mishandled relative imports inside a package's
`__init__.py` (resolving `from .seeding import …` to `tinygpt.seeding`), *and*
`seeded_generator` really was near-dead, imported but never called while `_data/batching.py`
hand-rolled the same generator. Both were fixed; edge count went 64 → 89.

One prediction was wrong and is left on the record: s2 was expected to also fail
`repo_no_pad_probability_mass`. It does not. The task still fails on two other checks.

---

## 2. Exhaustive reading is impossible by construction

The harness permits one tool call per turn, so an agent cannot issue more `read_file` calls
than it has turns.

```
files in repo                 48
max turns observed            24
structural ceiling on frac    0.5
exhaustive read possible?     False
```

This is a property of the budget, not of the model's restraint, and it is emitted as a
computed number so that changing either the turn budget or the repo size cannot silently
invalidate the premise.

Observed, well under the ceiling:

```
fraction of repo inspected    mean 0.181, max 0.271
every episode < 1.0           True
```

At most 13 of 48 files were read in any episode. The contract's machine check
(`fraction_repo_inspected < 1.0` for every episode) holds with a wide margin.

---

## 3. Results — 12 precommitted trajectories

`deepseek-chat`, 24 turns, 4 episodes per task. No re-rolls, no retuning.

```
located the relevant file     12/12
edited a relevant file         9/12
hardened pass (full fix)       9/12
exactly matches gold           9/12
naive pass (visible only)     12/12
naive false positives          3/12
mean turns                     19.3
```

| task | inspected | located | fixed |
|---|---|---|---|
| `s1_causal_mask_offbyone` | 0.146 | 4/4 | **4/4** |
| `s2_padding_supervised_as_labels` | 0.208 | 4/4 | **4/4** |
| `s3_warmup_offbyone` | 0.188 | 4/4 | **1/4** |

**Localization at 48 files is not the bottleneck.** The relevant file was opened in 12 of
12 episodes while reading under a fifth of the repo. Whatever limits performance here, it
is not the search space.

**s3 is the interesting failure.** All four episodes opened `_optim/schedule.py`, and three
still failed. Two of them edited `_train/loop.py` instead — the training loop — and one
edited nothing. The symptom ("the first optimizer step appears to accomplish nothing, and
the LR curve looks shifted by one step") is equally consistent with a
`scheduler.step()`/`optimizer.step()` ordering bug, which is a real and common defect and
lives in the loop. The model reached a plausible diagnosis, in the right neighbourhood, and
repaired the wrong thing. *Reading the file is not the same as reading it correctly.*

**Three naive false positives.** The visible suite passed in 12/12 episodes while the
hidden suite passed in 9 — the same visible/hidden gap this project measures everywhere,
reproduced at repo scale.

---

## 4. Corrections and limitations

**A measurement bug was found after the run and fixed without re-running (R17).**
`repo_fingerprint` hashed only `tinygpt/*.py`. For a flat Tier M package that is correct;
for Tier S it made every edit below the top level invisible, so the first pass reported
`edited a relevant file 0/12` alongside `hardened pass 9/12` — a contradiction, since you
cannot repair a defect without touching the file holding it. That contradiction is what
exposed it.

The rewards were never affected: they come from grading the workspace on disk, which never
consults the fingerprint. Only the patch metrics were wrong, and they were recovered
exactly from the graded workspaces, which still existed. `scripts/tier_s_recompute_patch_metrics.py`
spends no trajectories and preserves the original values in `*_ORIGINAL_BUGGY` fields so
the error stays auditable.

**The s1 symptom names the attention subsystem.** It says the team "refactored the attention
internals last sprint and split them across modules". That does not name the file or the
root cause, and it is how a real report reads — but it narrows 48 files to a neighbourhood,
and s1's 4/4 should be read with that in mind. The spec was frozen and the trajectories
were in flight when this was noticed; retuning the task after the fact is exactly what the
contract forbids, so it stands and is disclosed instead.

**Twelve episodes is small.** Per-task n=4. The s1/s2 versus s3 gap (8/8 against 1/4) is
large, but nothing here supports a significance claim and none is made.

**Localization is measured as "the relevant file was opened".** It does not measure whether
the agent understood the file, and s3 shows the difference is real.

---

## 5. Reproduce

```bash
python scripts/build_tier_s_template.py
docker run --rm -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/tier_s_freeze.py      # must print frozen=True

python scripts/run_deepseek_repo_eval.py --tier S --model deepseek-chat \
  --episodes 4 --max-turns 24 --out artifacts/tier_s_primary.jsonl
python scripts/tier_s_report.py
```

The freeze must succeed before the eval. It exits non-zero and prints
`FREEZE FAILED -- do not spend trajectories against this spec` otherwise.
