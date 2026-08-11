# PROTOCOL_CHANGELOG

Every change made after a real-model run had started, with the evidence for why
it is a **measurement** fix and not a task redesign. The rule this file exists to
enforce: results are never silently improved by editing the thing being measured.

Baseline frozen in `EXPERIMENT_MANIFEST.json`, label `tier_m_primary_v1`,
2026-08-10T07:22:11Z, 27 files hashed.

---

## R1 — Adapter smoke test exposed two defects (2026-08-10, before the primary run)

The DeepSeek tool-call adapter had never spoken to the live API. A single-episode
smoke test on `m1_attention_regression` with `deepseek-v4-flash` was run
deliberately *before* the 20-trajectory primary run, precisely so that adapter
defects would not contaminate it.

**Classification: INFRA_FAILURE / measurement bug. Both fixed before any primary
trajectory was collected, so no primary result is affected.**

### R1.a — `files_changed` reported the injected bug, not the model's edit

- **Observed:** the smoke episode reported `files_changed=['tinygpt/attention.py']`
  even though its only `apply_patch` call *failed*.
- **Cause:** the metric compared the workspace against the **gold** repo. The
  buggy workspace always differs from gold in exactly the mutated file, so the
  metric was reporting where the bug was injected.
- **Why it is a measurement bug:** it says nothing about the task or the model; it
  is a wrong computation of a logged field.
- **Fix:** snapshot `repo_fingerprint(ws)` *before* the episode and diff against
  that. Two fields now recorded: `files_edited_by_model` (vs. the pre-episode
  snapshot) and `files_still_differing_from_gold` (vs. gold).
- **Cherry-picking risk:** none. The old field was uninformative in both
  directions.

### R1.b — Valid patches rejected for a missing `@@` header

- **Observed:** at turn 10 the model produced the *correct* fix for the causal
  mask and the harness rejected it: `patch did not apply: no @@ hunk headers
  found`. The model had emitted a bare `@@` with no line numbers, then correct
  `-`/`+` lines.
- **Cause:** `parse_unified_diff` required a full `@@ -a,b +c,d @@` header.
- **Why it is a protocol-parser bug and not a difficulty reduction:** the
  submission contained the complete, correct edit. Rejecting it measured
  transcription formatting, not whether the bug was understood. This is the same
  category as the Phase 0.5 context-fuzz fix, where 3 of 4 "invalid" patches
  turned out to be our applier being stricter than `patch(1)`.
- **Fix:** `_parse_headerless()` accepts a bare block of `-`/`+`/context lines,
  with or without a content-free `@@` marker, as a single hunk located by content.
  File headers, `index` lines and the no-newline marker are skipped.
- **What did NOT change:** prose with no diff markers is still rejected; a `-`
  line that does not exist in the file is still rejected; no-op patches are still
  rejected. Verified by four dedicated cases plus the existing 24 patching tests
  (`tests/test_patching.py`, all passing).
- **Cherry-picking risk:** the change can only convert INVALID into a real grade,
  and the grade is then decided by the hidden suite. It cannot turn a wrong fix
  into a passing one.

**Guardrail reference:** G7 (protocol friction must not be scored as capability);
consistent with the Phase 0.5 finding that our applier, not the model, caused 75 %
of "invalid" patches.

---

## R2 — Primary evaluation configuration

Recorded for reproducibility, not a change:

| | |
|---|---|
| model | `deepseek-v4-flash` (cheaper of the two available; per instruction) |
| episodes | 4 per task × 5 Tier M tasks = 20 |
| max turns | 14 |
| temperature | 1.0 |
| max_tokens | 16,000 per call |
| grading | Docker sandbox, `--network=none`, read-only rootfs |
| output | `artifacts/tier_m_primary.jsonl`, log in `artifacts/raw/` |

The smoke episode is retained separately as `artifacts/smoke.jsonl` and is **not**
merged into the primary results — it ran under the pre-R1 protocol.

---

## R3 — the independent oracle was weaker than the verifier it audits

Found **after** the Tier H evaluation, while reading the two trajectories where
the hardened reward and the independent audit disagreed.

**Classification: measurement bug in the AUDIT ORACLE. No reward changed. All
audits regenerated with the corrected oracle; no trajectory was re-sampled.**

### What happened

The first Tier H audit reported two disagreements on `h3_accumulation_and_clipping`
(e1, e3): hardened said 0, the independent probe said `FULL_FIX`. Read as-is that
would have been this project's first hardened false negative on real data.

It was not. `h3` e1 has `files_edited_by_model = []` — **the model made no edits at
all** in 14 turns. The buggy accumulation and the clip-after-step were both still
present. `hardened=0` was correct and the probe was wrong.

### Why the probe was wrong

`independent_truth()` compared: logits against gold, supervised token counts, final
loss after 24 steps within a tolerance band, and the LR trace. `h3`'s defects live
entirely in `accumulate_gradients` and in the order of `clip_grad_norm_` versus
`optimizer.step()`. None of those four comparisons can see either one:

- logits are unaffected (the bug is in the loop, not the model);
- token counts are unaffected;
- the LR trace is unaffected;
- final loss after 24 steps stayed inside the tolerance band.

So the oracle was structurally blind to the entire F4 family. This is the same
class of limitation already recorded for `C2` in `VERIFIER_FUZZ_AUDIT.md` — the
probe cannot see optimizer group structure — but here it produced a **wrong label
on real data**, which the C2 case did not.

### Fix

Two invariants added to `independent_truth()`, both computed inside the probe and
neither taken from the candidate's module:

1. **Accumulation equivalence** — sum the per-batch loss sums, divide once by the
   global supervised-token count, single backward, compare gradient vectors.
   Mathematically identical to a full-batch backward and does not require padding
   batches to a common width.
2. **Clipping effectiveness and non-zero gradients** — instrument
   `AdamW.step` and record the gradient norm actually present at each update.

### Verification

- Fuzz audit re-run: all 13 probes unchanged, `D2/D3/D4` (genuine FULL_FIX) still
  pass, `C2` remains the single known probe disagreement.
- Tier H re-audited: **0 disagreements** (was 2). `h3` e1 → `WRONG`, e3 →
  `PARTIAL_FIX`, both matching hardened.
- Tier M and the pro confirmatory run re-audited with the same oracle for
  consistency: **no label changed** in either.

### Why this is not cherry-picking

The change made the audit **stricter**, not more permissive: two trajectories moved
from `FULL_FIX` to `WRONG`/`PARTIAL_FIX`, lowering the reported Tier H success rate
from 10/20 to 8/20. A revision that lowers your own headline number after you have
seen it is the opposite of tuning. The pre-fix labels are recoverable from
`PROTOCOL_CHANGELOG` plus the raw JSONL, which was never modified.

### What it says about the methodology

Guardrail G3 warns that a verifier is only a proxy. This is the same warning
applied one level up: **an audit oracle is also only a proxy, and needs auditing.**
"Independent" bought genuine independence from the verifier's fixtures, and bought
nothing at all about coverage. Recorded in `VERIFIER_QUALITY_MATRIX.md` as horizon
item H3.


## R4 -- contract check narrowed after the replay exposed it as over-strict

Found by the G2 replay, i.e. **after** the verifier was written and after results
were seen. Disclosed here in full because that is the situation this file exists
for.

**Classification: over-strict verifier, corrected. The change makes v2 LESS strict,
so it cannot manufacture a better false-positive rate.**

### What happened

The first replay reported `v2_FN_count = 1`: `m5_masking_interaction` e1, which the
independent oracle labels `FULL_FIX` and v1 accepted, was rejected by v2 on
`repo_contract_return_types` with *"both loss_sum return values must be Tensors"*.

Contract criterion 7 exists precisely to catch a verifier that lowers FPR by
rejecting legitimate fixes, so this was recorded as a **FAIL** before being
investigated.

### Ruling out the cheap explanation first

The row was recovered by `RECONSTRUCTED_FROM_RAW`, so the reconstruction itself was
the obvious suspect. `scripts/test_reconstruction_fidelity.py` discriminates: it
re-runs the **oracle** on the reconstruction and compares against what the original
audit recorded on the live workspace.

```
original audit (live workspace): label=FULL_FIX semantic_ok=True contract_ok=True
oracle on reconstruction       : semantic_ok=True contract_ok=True
v2 contract layer              : repo_contract_return_types = False
```

The reconstruction is faithful. The oracle and v2 disagree **about the same bytes**,
so this was a genuine verifier problem, not a recovery artifact.

### Root cause

The model's fix returned the supervised-token count as a Python `int`
(`int((shift_labels != ignore_index).sum())`) where gold returns a 0-dim `Tensor`.
Semantically identical; every call site immediately coerces it.

`VERIFIER_V2_PROTOCOL.md` states that contract checks cover *documented interfaces,
not taste*, and explicitly excludes implementation detail. Requiring `Tensor` rather
than `int` for a scalar count is implementation detail. **The check was wrong, not
the fix.**

### Fix

`repo_contract_return_types` now requires:

- `loss_sum[0]` **must** be a `Tensor` — `backward()` depends on it, so this is a
  real interface requirement;
- `loss_sum[1]` may be `int` **or** `Tensor`, and must be a non-negative integer
  scalar.

Nothing else changed. The Gap-1 requirement (`accumulate_gradients` returning
`float`) is untouched, because a `Tensor` there silently changes downstream
arithmetic and logging and *is* load-bearing.

### Verification after the change

| | before | after |
|---|---|---|
| gold passes contract layer, 10/10 repo tasks | yes | yes |
| `v2_FN_count` | **1** | **0** |
| v1↔v2 disagreements | 1 | 0 |
| v1 FP rate / v2 FP rate | 0.2778 / 0.2817 | **0.2778 / 0.2778** |
| replay coverage | 98.9 % | 98.9 % |
| criterion 4 / 6 / 7 | False / True / False | **True / True / True** |
| `D1_m3_return_tensor` (Gap 1): v1 → v2 | — | **PASS → FAIL, as required** |
| v2 rejections of FULL_FIX probes | — | **0** |
| A/B/C exploit probes still caught by v2 | — | **9/9** |

### Why this is not moving the goalposts

The revision made the verifier **less** strict. A less strict verifier can only
raise its false-positive rate, never lower it — and the measured FP rate was
unchanged at 0.2778 while the false negative went to zero. The Gap-1 closure is
independently demonstrated by `D1`, which v1 accepts and v2 rejects. If this change
had been self-serving it would show up as FP going down, and it did not.
## R5 -- h2's "too many unusable actions" was an adapter defect, not difficulty

Found while root-causing the Tier H `h2_position_double_defect` 0/4 result, as the
foundations work required before spending more paid trajectories.

**Classification: TYPE A harness/adapter defect. It caused a scientific
misreading, which is why it is recorded here and not just fixed.**

### Evidence

Both 24-turn `h2` episodes investigated normally, then died:

```
h2 e0: list_files 1, read_file 11, run_command 4, then noop x3  -> ended turn 19
h2 e1: list_files 1, read_file 10, run_command 3, then noop x3  -> ended turn 17
both: edited=[]   api_errors=[]   end=too many unusable actions
```

Every `noop` carried `{'text': ''}` -- the provider returned an assistant message
with **neither content nor a tool call**.

### Root cause, and why it was always exactly three

The adapter echoed that contentless message straight back into the history
(`msg.model_dump(exclude_none=True)` on an empty message yields little more than
`{"role": "assistant"}`). A history containing a contentless assistant turn then
makes the *next* completion come back empty too. So one empty response
deterministically produced an unbroken run of them, and `run_episode`'s
three-strikes guard ended the episode. The guard was working; it was being fed
adapter noise.

Compounding it: `_dispatch` returns `None` for the adapter's synthetic action names
(`noop`, `api_error`, `bad_arguments`), so a **provider failure counted as the
policy emitting an invalid tool** -- conflating INFRA_FAILURE with model behaviour,
which the failure taxonomy explicitly separates.

### Fix

1. `DeepSeekPolicy` no longer echoes a contentless message. It appends a
   placeholder plus an explicit corrective user turn and returns
   `Action("empty_response")`, so the conversation stays well-formed.
2. Prose-without-a-tool-call is handled separately as `no_tool_call`, keeping the
   content and restating the protocol requirement.
3. `run_episode` now has `ADAPTER_NONACTIONS = {api_error, empty_response,
   no_tool_call, noop}`. These are recorded in the trace but do **not** consume the
   unusable-action allowance and do not end the episode.
4. The three-strikes guard still fires for genuinely invented tool names.

### Regression tests (`tests/test_harness_protocol.py`)

- five consecutive empty responses no longer end the episode;
- after three empties the policy still makes progress and its tool call executes;
- an invented tool name **still** ends the episode after exactly three;
- `ADAPTER_NONACTIONS` must name every synthetic the adapter can emit.

### Scientific consequence

`h2`'s 0/4 at 14 turns and 0/2 at 24 turns are **not capability measurements**.
`TURN_BUDGET_ABLATION_REPORT.md` already records `h2` as
`UNMEASURED - protocol failure at both budgets`; that label now has a named root
cause. `h2` must be re-run under the fixed adapter before it carries any
difficulty claim. No trajectory was relabelled to look better -- the correction
removes a data point rather than adding a success.

---

## R5b -- G0 did not index the graded workspaces

Same investigation, second defect.

`scripts/build_sandbox.py` used `./.sandbox_work` as its scratch directory and
`shutil.rmtree`s it. Phase 1's final verification ran that script *after* the Tier
M evaluation and silently deleted all 20 graded workspaces. G0 was designed to
prevent exactly this and missed it, because it hashed JSONL and audit files but not
workspaces.

**Fix**

1. `build_sandbox.py` now uses `.sandbox_selftest`, with the reason in a comment.
2. G0 indexes the preserved per-trajectory source trees
   (`artifacts/raw/*_final_sources/`) file by file -- these are the reconstruction
   source of record. The manifest went from **56 to 127 artifacts**, still 0
   missing, still 29/29 metrics regenerating.
3. An AST-based test forbids any script from passing an evidence directory to
   `rmtree`. It resolves one level of variable indirection so that *reading* an
   evidence path (as the replay legitimately does) is not flagged, and a companion
   test proves the detector fires on the original defect and stays quiet on the
   legitimate pattern.

---

## R6 -- numpy scalars passed the float contract (E2)

Found by implementing E2 from the frozen `VERIFIER_V2_PROTOCOL.md`.

**Classification: the check did not implement its own frozen specification.**

E2 was specified before the contract check was written: *"train() returns
final_loss as a numpy scalar ... numerically identical, contractually wrong, and
the kind of thing that silently breaks JSON logging downstream."* Expected v1
accept, v2 reject.

Measured: **v2 accepted it.** Cause is a Python subtlety --
`numpy.float64` subclasses `float`, so `isinstance(np.float64(1.0), float)` is
`True`.

**Fix:** the two documented-float returns now use `type(x) is float` rather than
`isinstance`, and `train()`'s history must additionally survive `json.dumps`,
which is the actual downstream breakage E2 describes.

**Verification:** gold still clean on 10/10 repo tasks; all three E probes now
rejected by v2 and accepted by v1; v2 rejects **0** FULL_FIX probes; the
89-trajectory replay is unchanged at `v2_FN = 0` and `v2_FP == v1_FP = 0.2778`.

**Why this is not goalpost-moving:** the probe predates the check. Tightening the
check to satisfy a criterion frozen before implementation is implementing the
specification. The replay confirms it cost nothing in false negatives.
---

## Retained failures

Nothing is deleted. Every run kept, including superseded ones:

| artifact | protocol | status |
|---|---|---|
| `artifacts/smoke.jsonl` | pre-R1 | retained, excluded from analysis |
| `artifacts/tier_m_primary.jsonl` | post-R1 | primary result |
| `artifacts/deepseek_baseline.jsonl` | Phase 0.5, Tier E single-turn | retained |
| `artifacts/deepseek_flash_baseline.jsonl` | Phase 0.5, Tier E single-turn | retained |
