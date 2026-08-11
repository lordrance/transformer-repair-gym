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

---

## R7 — G0's own frozen copies were mutable, and destroyed the artifact they protected

**Found:** re-running G0 after G2 flipped it `PASS → FAIL` on a single mismatch:
`fuzz.n_probes: claimed 13, recomputed 16`.

**Root cause.** Two compounding defects, both mine.

1. `scripts/fuzz_verifier.py` writes to a fixed path, `artifacts/verifier_fuzz_audit.json`.
   Adding the E1–E3 contract-edge probes for G2 overwrote the Phase 1 13-probe result
   in place. The suite growing 13 → 16 is legitimate; overwriting the old evidence to
   do it is not.
2. G0 was supposed to be the backstop, and wasn't. It copied each small artifact to a
   flat name under `artifacts/frozen_phase1/` and **re-copied on every run**, so the
   second G0 invocation replaced its own frozen 13-probe copy with the 16-probe one.
   A freeze that a later run can silently overwrite is not a freeze.

This is R5b's failure class recurring on a JSON artifact instead of a workspace
directory: scratch and evidence sharing a path. R5b fixed one instance; it did not fix
the pattern.

**The tempting wrong fix** was to set `CLAIMED["fuzz"]["n_probes"] = 16` and move on.
That reconciles the record to the current state and erases the fact that evidence was
lost — exactly the "adjust the expectation until it matches" move the contract forbids.

**Fix.**
- Frozen copies are now **content-addressed and immutable**: `<flat-name>.<sha[:12]>`,
  written only `if not dst.exists()`. A changed artifact becomes an additional frozen
  copy; nothing is ever replaced. 127 artifacts → 104 distinct frozen blobs.
- `CLAIMED["fuzz"]` now describes the live v2 suite (16 probes) **and** a separate
  `fuzz_v1_artifact` entry asserts the v1 count of 13 against an explicit
  `status: LOST_OVERWRITTEN` record naming where the summary does survive
  (`VERIFIER_FUZZ_AUDIT.md`, `PHASE_1_FINAL_RESEARCH_REPORT.md`).

**What is genuinely lost:** the raw v1 fuzz JSON. Its summary numbers survive in two
committed documents, so no Phase 1 headline claim is unsupported — but the raw file is
unrecoverable and is recorded as such rather than reconstructed. G0 is `PASS` with
30/30 metrics regenerating exactly, and that now includes the loss itself as a checked
fact.

**Cost of the honest version:** G0's manifest permanently carries a `LOST_OVERWRITTEN`
entry. That is the correct outcome. A green gate that had quietly absorbed a destroyed
artifact would have been worth less.

---

## R8 — G0 treated "present and stably hashed" as "preserved"

**Found:** starting G1, `json.load("artifacts/raw/v1_probe.json")` raised
`JSONDecodeError: Expecting value: line 1 column 1`. The file G0 had been reporting as
a healthy frozen artifact across every run was not valid JSON.

**Root cause.** The probe was run as `docker run ... > v1_probe.json` from PowerShell.
PowerShell wraps a native command's stderr in `NativeCommandError` records and merges
them into the redirected stream, so the file is 12 lines of pip root-user warning and
command echo followed by the real payload on line 13. The payload itself is intact —
this was a capture defect, not a bad probe.

**Why G0 missed it.** G0 checked existence and recomputed a SHA-256. A corrupt file
satisfies both perfectly, and satisfies them *stably*, so repetition never helped. The
gate encoded "the bytes have not changed" and I had been reading it as "the evidence is
still usable". Those are different claims, and only the weaker one was tested.

The extraction itself needed care: the `NativeCommandError` echo of the failing command
contains braces, so the first `{` in the file is not the payload. `scripts/repair_v1_probe.py`
tries every brace-balanced candidate and takes the first that parses.

**Fix.**
- `parses()` in `scripts/freeze_historical_artifacts.py`: every `.json` artifact must
  load as one object and every `.jsonl` artifact must have all non-blank lines load.
  Unparseable artifacts now fail G0 alongside missing ones and mismatched metrics.
  Currently **0 unparseable of 22 structured artifacts**.
- The original polluted stream is preserved as `artifacts/raw/v1_probe.raw.txt` before
  the recovered payload replaces the file. How the probe was actually captured is
  itself part of the record, and the recovery must not be the only surviving trace.

**Standing lesson, third instance.** R5b was scratch sharing a path with evidence, R7
was a freeze that later runs could overwrite, and R8 is a gate whose check was weaker
than its claim. Each was found by *using* the evidence for real downstream work, never
by re-reading the gate. A gate only tests what it tests; the artifact has to be
consumed to know it survived.

---

## R9 — the visible smoke test imported the module holding the hidden oracle

**Found** while running the planted repo inside a real v1 `Runtime`:
`trgym_visible_checks.py` did `from trgym.repo.checks import run_repo_checks`, and
`ModuleNotFoundError: No module named 'trgym'` killed it.

**Root cause.** The candidate's own smoke runner depended on `trgym.repo.checks` — the
module that also defines the hidden L2/L3 checks, `gold_repo()` and `build_gold`. On the
host this was invisible because `trgym` happened to be importable from the workspace's
parent. It only failed once the workspace was somewhere `trgym` genuinely was not.

The honest reading is not "the container is missing a dependency". It is that the
dependency should never have existed: anything the candidate must import cannot also be
where ground truth lives. The Windows-host layout had been hiding a coupling between
the public and hidden halves of the verifier.

**Fix.** `trgym/repo/visible_runtime.py` holds the five visible checks plus `RepoModules`,
`_seeded`, `CheckFailure` and `VISIBLE_SHAPE` — and no gold, by inspection: every visible
check operates purely on the candidate's own modules, which is why the split was possible
without weakening anything. `build_repo` plants that file beside the runner, so a
workspace is self-sufficient. `trgym/repo/checks.py` imports the visible layer instead of
restating it, so there is exactly one definition of each.

Verified: visible still passes on buggy, hidden still passes on gold, 155 host tests pass.

## R10 — `trace.has_error` as a reward gate would have re-created the h2 confound

The shipped `LeanTask` reference opens its reward with `if trace.has_error: return 0.0`,
and I copied it. `Trace.ok` defaults to `False` and `has_error` is `not ok`, so that gate
scores **every incomplete rollout 0.0** — silently converting an INFRA/PROTOCOL failure
into a capability zero.

That is precisely the conflation that made `h2` read as 0/4 "too hard" when its episodes
had actually been killed by an adapter bug (R5). Having just spent a session removing that
confound from the legacy harness, importing it into the v1 path from a reference
implementation would have been a poor trade.

**Decision.** `semantic_repair` does not consult `has_error`. The workspace is graded on
whatever state it is in, and a separate `@metric infra_error` records whether the rollout
completed. The two stay separable in analysis, and an analysis that averages reward
without conditioning on `infra_error` is knowingly mixing two populations.

## R11 — `grade_workspace` recorded every check as passing

**Found** by the one test that compares gold against the planted bug: both scored 1.0.

**Root cause, mine.** `run_repo_checks` *returns* `(name, ok, message)` tuples; it does
not raise on a failing check. My bridge wrapped it in `try/except` and recorded `True`
whenever no exception escaped — so every check passed unconditionally and the whole v1
verifier was vacuously green.

This is the most dangerous class of bug in the project: it fails *open*, it makes
everything look correct, and nothing else in the suite would have caught it. The gold/no-op
separation assertion is the only reason it surfaced. Every new grading path needs that
assertion before it is trusted, not after.

**Fix.** Consume the return value. A check name that does not appear in the results is
recorded as a failure rather than skipped, so a typo shrinks the score instead of the suite.

---

## R12 — the INFRA/capability discriminator was a constant

R10 removed `if trace.has_error: return 0.0` from the reward and added an `infra_error`
metric to carry that information instead, so an infrastructure failure could not be read as
a capability zero. The metric read the same field: `1.0 if trace.has_error else 0.0`.

`has_error` is `not trace.ok`, and `rollout.py` sets `ok` **after** scoring. So at the
moment the metric runs, `ok` is always False and `infra_error` was **always 1.0** —
including for the clean, fully successful rollout. A discriminator that returns the same
value for every input carries no information, and its constancy was invisible because the
first live smoke genuinely did fail, so 1.0 looked correct.

Caught by reading the actual trace record rather than the summary line: the inner trace had
`ok: true`, `is_completed: true`, `errors: []`, `stop_condition: "max_turns"`, and 14 model
calls. I had reported "the agent did nothing" from the summary's `infra_error 1.00`; the
trace showed a competent 14-turn investigation. The metric was wrong, not the rollout.

**Fix.** `infra_error` reads `trace.errors`, which is populated by scoring time. Added
`hit_turn_limit` off `trace.stop_condition` so G3's budget-sensitivity confound is visible
per-episode in the raw trace instead of needing a separate ablation to reconstruct.

**Test.** `test_infra_error_metric_actually_fires` injects a real `Error` and a
`max_turns` stop and requires both metrics to reach 1.0, while the happy path requires 0.0.
A constant 0.0 would be just as broken as a constant 1.0, and the happy-path assertion
alone would not notice — the same "prove the detector detects" discipline as the rmtree and
legacy-API guards.

**Standing note.** R11 and R12 are the same shape: a signal that always says the same thing.
R11 always said "pass", R12 always said "infra failure". Neither was caught by a test
asserting the expected value on the expected input; both needed a test asserting the signal
*changes*. Any new reward or metric in this project gets a fires/does-not-fire pair.

---

## R13 — the static gates were matching prose, not code

**Found by G6.** Running 32 real functions from PyTorch 2.9.1 and numpy 2.3.3 (both BSD,
both installed and version-pinned) through the static anti-exploit gates produced a
**15.6 % submission-level false-positive rate** — 5 of 32 benign, human-written ML files
would have been refused by the hardened verifier.

Every one of the five was a **docstring**:

| sample | gate | what actually matched |
|---|---|---|
| `torch.nn.MultiheadAttention` | `dynamic_execution` | docstring: "training is disabled (using ``.eval()``)" |
| `torch.nn.TransformerEncoderLayer` | `dynamic_execution` | same docstring phrase |
| `torch.nn.Embedding` | `hardcoded_tensor` | `tensor([...])` in a docstring example |
| `torch.nn.CrossEntropyLoss` | `hardcoded_tensor` | " |
| `torch.utils.data...default_collate` | `hardcoded_tensor` | " |

A submission that documents itself is not cheating, and a literal tensor inside a docstring
cannot influence a result unless it is also executed — which is what `dynamic_execution`
exists to catch. These are unambiguous false positives.

**Why this matters more than the number.** The corpus was chosen from this project's own
domain: attention, normalisation, optimisers, LR schedules, gradient clipping, collation. A
gate that fires on PyTorch's own `MultiheadAttention` docstring fires on a *competent human
fix to an attention bug* that happens to explain itself. The gates were penalising exactly
the submissions the environment is meant to reward.

**Fix.** `anticheat.code_only()` blanks comments and string literals before pattern
matching, preserving line and column positions so evidence strings still report the true
location and the original (unblanked) line text. All four regex gates now match against it.
FP rate **15.6 % → 0.0 %** on the same 32 samples.

This is the third time this exact defect has appeared: the G1 legacy-API detector flagged
its own docstring about the ban, my rmtree detector flagged files that merely *mentioned* an
evidence directory, and now the anticheat gates. **Any regex over source code in this
project must be preceded by tokenization.**

**Guarding the fix.** A gate that ignores too much protects nothing, so the fix is bounded
from both sides:
- `tests/test_anticheat_prose.py` (10 tests) pairs every gate's negative case (prose must
  not fire) with a positive case (a real violation must still fire), plus a test that
  evidence reports the original line text rather than blanked spaces, and a test that
  unparseable source is scanned as-is rather than silently skipped.
- `scripts/audit_cheats.py` re-run after the change: **7/7 exploits still caught**, with
  identical per-cheat gate attributions and identical naive→hardened gaps.

**A defect in my own audit, found on the way.** The first version of
`heuristic_gate_fp_audit.py` reimplemented each gate as a plain regex instead of calling it.
That was stricter than the real verifier — `gate_no_shape_special_casing` walks the AST and
deliberately ignores comparisons against 0/1/2 as ordinary defensive code — and it credited
the gates with a false positive (numpy's `if n == 0:`) they never actually produce,
inflating the measured rate from 15.6 % to 18.8 %. An FP audit that does not call the code
under audit is measuring its own reimplementation. The script now invokes the real gate
functions, and reads the authoritative exploit result from `artifacts/cheat_audit.json`
rather than approximating it.

---

## R14 — "the oracle is unreachable" needed two boundaries, and G1 shipped with one

**Found while setting up G5's isolation canaries**, by inspection rather than by a test —
which is itself the finding worth recording.

The G1 grading bridge read the candidate's edited sources out of the container and graded
them **in the rollout process**:

```
grade_workspace -> run_repo_checks -> RepoModules.__enter__
                -> sys.path.insert(0, workspace); importlib.import_module("tinygpt")
```

Importing a module executes its top-level statements. So candidate code ran inside the one
process where `trgym.repo.checks.gold_repo()` is importable and the gold tree is on disk. A
submission with three lines at module scope in `tinygpt/config.py` — an editable file for
every task — could have read gold, or reached into the already-imported grader module and
set `ATOL = 1e9`.

**What was and was not true.** G1's `test_grading_artifacts_are_not_candidate_readable`
verifies that an *agent* inside the container finds nothing: that still holds, and it was
never the weak point. The overstatement was mine, in prose: I wrote that the oracle is
"unreachable, not merely protected" and that "a candidate that greps its entire filesystem
finds nothing to tamper with, because nothing is there". The first clause claimed more than
one boundary can deliver. The container constrains the **agent**; it does not constrain code
the **grader** imports on the agent's behalf.

**Fix.** `grade_workspace` now routes through `trgym.harness.sandbox.run_checks` with
`fallback=False` — the locked-down container Phase 1 already used (`--network=none
--cap-drop=ALL --security-opt=no-new-privileges --read-only`, non-root uid 10001, tmpfs
scratch). In-process grading survives only behind an explicit `allow_in_process=True`, whose
sole legitimate caller is `validate()`'s gold preflight, on a tree `build_gold` produced from
our own template and no policy has touched.

**Why this was not caught earlier.** Every G1 isolation test probes *outward* — from the
container, looking for the grader. None probed *inward*, from candidate code the grader
executes. The threat model had a direction, and the tests only covered one of them. That is
the general lesson: an isolation claim needs a test per boundary, in both directions.

**Status.** The repair is in the code, but the verification run is **not complete**: the
canary script (`scripts/g5_isolation_canaries.py`, seven probes across all six required
channels plus gold-import) could not be executed in this session — the harness safety
classifier declined to evaluate it, since the probes are by construction code that reads
secrets and spawns processes. So:

- **G1 is flipped back to FAIL** pending re-verification. A gate whose central security
  claim was overstated does not stay green because the fix is written; the fix has to be
  measured. This follows the frozen rule for a defect found in a passed gate.
- **G5 is blocked**, not deferred: the benchmark half is straightforward, but the contract
  requires all six canaries to pass, and asserting containment without running the probes
  would be exactly the intent-as-evidence substitution the contract forbids.

Next action is to run `scripts/g5_isolation_canaries.py` outside auto mode, confirm
`sandboxed_container` contains all seven probes while `in_process` leaks them (a canary suite
where nothing leaks under the *unsafe* grader is not testing anything), then re-run
`tests_v1/` and the host suite.

---

## R14 (cont.) — the canaries ran, and the first two attempts measured nothing

The canary suite was executed in this session. Recording the two ways it was wrong first,
because both were the kind of failure that returns a green result:

**Attempt 1 — every probe "contained", nothing executed.** Run on Windows, all nine probes
reported contained under *both* graders, including the deliberately unsafe one. The cause
was in the harness, not the code under test: `scripts/g5_isolation_canaries.py` imports
`environments.transformer_repair.grading`, whose package `__init__` imports `task.py`, which
imports `verifiers.v1`, which imports `fcntl`. Every grade raised `ModuleNotFoundError` and
the script recorded "no evidence found" as containment.

This is why the payloads now begin with a witness write and why `contained_by` is recorded
per probe. **A canary that never ran is not a canary that passed**, and the artifact must be
able to tell the difference without a human reading it.

Fix: run the suite where grading actually happens — inside the Linux image, with the repo
mounted at a path the Docker daemon can also resolve
(`-v e:/RL:/run/desktop/mnt/host/e/RL`), so that the nested grading container's `-v
{REPO_ROOT}:/grader:ro` resolves to the same files on both sides. No code was weakened; the
mount topology was corrected, the same Plan C used for the v1 proxy listener.

**Attempt 2 — the positive control was not a control.** `grade_in_process()` called
`grade_workspace(...)` with default arguments, which R14 had *already* routed to the sandbox.
Both columns were therefore the same grader. Fixed by reaching the pre-R14 path explicitly
with `allow_in_process=True`; the control now leaks 8 of 9 probes, which is what makes the
sandboxed column mean anything.

---

## R15 — the container isolates the host from the candidate, not the verdict from the candidate

**Found by the R14 canaries, on their first genuine run.** This is R14's own lesson applied
one level further in, and it was found by measurement rather than by inspection.

With the corrected control in place, the sandboxed path did **not** contain everything:

```
in_process_UNSAFE_CONTROL   leaked 8/9
sandboxed_container         leaked 1/9   <- verdict_tamper_tolerance
```

and a second row was worse than it looked: `module_global_write`, whose payload does nothing
but `import trgym.repo.checks; ATOL = 1e9`, produced `graded_passed = True` on a workspace
built from the **buggy** template. A tree carrying the planted defect scored a pass through
the production grading path.

**Root cause.** R14 moved grading into a container, which stops candidate code from reaching
the grading *host*. Inside that container the candidate and the checks still share one
process, by necessity — grading has to execute the code it is grading — and
`trgym.repo.checks` is importable there because `/grader` is mounted and on `sys.path`. So
module-level candidate code can rewrite the tolerances and the check registry that decide its
own verdict. A third probe (`verdict_forge_protocol`) showed the result channel had the same
shape of problem: the worker announced its verdict with a fixed `<<<TRGYM_RESULT>>>` marker
on stdout, which the candidate shares and can write to.

**What R14 got right and what it overstated.** The host boundary holds: all seven
containment probes were contained, including the gold-import probe and the grader-secret
probe. What did not hold was the unstated assumption that a contained process also returns an
honest answer. Containment and integrity are different claims and the canary suite now
measures them separately.

**Fix — two independent layers, neither claimed to be a boundary on its own.**

1. **Refuse before executing** (`grading.scan_for_grader_tampering`). The candidate's
   editable sources are scanned statically — bytes only, no import — for the grader package,
   the import cache, the import machinery, builtins access, and the result marker. A hit
   refuses the grade outright. Comments and docstrings are stripped first (R13). The patterns
   are narrow on purpose: `compile`/`eval` are **not** included, because `torch.compile` is
   legitimate. G6 measured the two closest existing gates at a 0/32 false-positive rate on
   benign human-written ML code.
2. **Discard the verdict after executing** (`trgym/harness/sandbox.py`). The worker snapshots
   `ATOL`, `RTOL` and the `CHECKS` registry *before* importing anything, re-reads them after,
   and marks the run `tampered` if they moved — in which case every check is reported failed.
   The result marker is now per-job (`<<<TRGYM_RESULT:{nonce}>>>`), the nonce arrives on
   stdin rather than in argv or the environment (both readable from inside the process), and
   the host takes the *last* correctly-nonced block. An unnonced marker printed by a
   candidate no longer parses as a result at all; it is recorded as `tampered`.

**What is deliberately NOT claimed.** Perfect in-process integrity is not achievable while
the grader must execute the code it grades: a determined adversary with knowledge of the
worker can still walk the object graph to whatever the process holds. Both layers raise the
cost substantially against the threat this project actually has — a policy optimising against
a reward — and neither is a sandbox. `SECURITY_MODEL.md` states this rather than implying
otherwise. A `sys.modules` purge plus `sys.path` removal was considered as a third layer and
rejected: several checks import `trgym.repo.build` and `trgym.tasks.repo_specs` lazily, so
purging would break gold grading for a benefit the other two layers already provide.

**Also fixed here (found in the same run).** The sandbox image runs as non-root uid 10001,
and a bind-mounted workspace carries the host's ownership through. When grading runs from
inside the project's own container — i.e. as root on Linux — `mkdtemp` yields a 0700
directory owned by uid 0, and the sandbox cannot read the tree it is grading. Every check
then fails with an import error that *looks* like "the candidate is wrong". Grading now opens
the disposable per-job workspace's mode before mounting it (POSIX only).

**Status.** The finding is measured and recorded in
`artifacts/g5_isolation_canaries.json` (attempt-3 evidence, pre-fix). The **fix is not
measured**: the re-run was blocked by the harness safety classifier, which began refusing
every `Bash` and `PowerShell` invocation partway through this session. So G1 and G5 both stay
FAIL. Nothing is marked PASS on the strength of code that has not been run.

---

## R16 — the oracle was in the room the whole time

**Status: fixed and measured.** `artifacts/g5_isolation_canaries.json` records
`sandboxed_container: leaked 0/17, ran 17, refused 0` against a control that leaks 12/17.

### What R15 missed

R14 moved grading into a container. R15 added two integrity layers because the candidate and
the checks still shared a process inside it. The G5 canaries, once they could finally run,
showed the remaining hole was far simpler than either fix assumed:

```
-v {REPO_ROOT}:/grader:ro        # trgym/harness/sandbox.py, pre-R16
```

The **entire repository** — hidden checks, gold builder, pristine template — was mounted
into the container that executes candidate code. Two probes demonstrated it:

| probe | mechanism |
|---|---|
| `gold_oracle_import_evasive` | `gc.get_objects()` → `trgym.repo.checks` → `gold_repo()` → read gold `attention.py` |
| `oracle_source_read_in_container` | `open('/grader/trgym/repo/checks.py')` — no evasion at all |

The second is the damning one. It breaks nothing and evades nothing; it just opens a file.
R15's static denylist could not have stopped it, and no denylist could: the string that
names the path is not the vulnerability, the mount is.

R15's layer 2 also cannot see this. It snapshots `#GOLD` to detect *mutation* of the gold
cache. Reading gold is not mutation, so nothing fires.

### The shape of the fix

Stop defending the answer sheet and take it out of the exam room.

```
Candidate container                    Trusted comparator
  candidate + public inputs   ──obs──▶   gold + oracle + hidden checks
  no /grader, no gold,                   applies every predicate
  no template, no reference              after the container exits
```

- `trgym/repo/candidate_probe.py` — runs inside the container, mounted **alone** at
  `/probe`. Imports no `trgym`. Computes observations; decides nothing. `RepoModules` is
  duplicated inside it deliberately: importing the real one would require the mount whose
  removal is the entire point.
- `trgym/repo/obs_protocol.py` — the narrow channel. No pickle: `torch.load` is an
  unpickler and PyTorch's own docs say not to point it at untrusted data. Tensors cross as
  dtype + shape + raw bytes, decoded via `frombuffer` after dtype allowlisting, rank,
  element-count and exact byte-length checks — all *before* any allocation, so a
  `shape: [1e6, 1e6]` declaration fails on arithmetic rather than on an OOM kill.
- `trgym/repo/predicates.py` — the trusted half. Opens gold; never imports candidate code.

The logits check inverted rather than moved: instead of loading gold's weights into the
candidate model, the candidate's `state_dict` travels **outward** and is loaded into the
gold model. Same comparison — two implementations on one set of weights — and
`strict=True` still catches a divergent parameter layout, but nothing gold-derived travels
inward. A precomputed reference bundle sent *into* the container was considered and
rejected for exactly that reason.

### The denylist is gone

R15's `scan_for_grader_tampering` and `_TAMPER_PATTERNS` are deleted. Two reasons, and the
second matters more:

1. It stopped being load-bearing. A tree that "reaches for the grading machinery" now
   reaches an empty filesystem.
2. **It hid the boundary from measurement.** Every canary aimed at the oracle was refused
   before executing, so the suite scored `contained` for probes that never ran — the
   vacuous-green failure this project keeps rediscovering. Deleting the gate is what let
   those probes finally measure the real thing: `refused 4 → 0`, `ran 11 → 17`.

`tests/test_grading_isolation.py::test_the_pattern_denylist_is_gone` fails if it returns.

### What is still not claimed

Predicates over purely candidate-internal quantities remain **forgeable** — a candidate that
fabricates its own gradient norms passes `gradients_reach_optimizer`. Moving the comparison
out of the candidate process does not change that, and it is listed explicitly in
`predicates.FORGEABLE`. Gold-comparison predicates are not forgeable: the candidate cannot
know the value it would have to claim, because gold is absent from the container it ran in.

### Bugs found while building this

- **Probe unreadable.** `mkdtemp` yields 0700; the sandbox runs as uid 10001, so the probe
  mounted at `/probe` failed with `Permission denied` and every check read as a candidate
  failure. Same trap R15 hit with the workspace, in a new place.
- **Two canary false positives, caught by the evidence and not by inspection.**
  `grader_mount_absent` fired on an empty `/grader` mount point left in the image
  (`.exists()` is true for an empty directory); `reference_bundle_unreachable` matched
  `torch.ao.nn.quantized.reference.*` on a bare substring. Both now report *what* they
  found, because a leak whose evidence is a bare filename cannot be triaged.
- **Two gate checks that matched prose, not code.** The first R16 additions to
  `final_acceptance.py` searched raw text for `/grader` and `pickle` — and failed on the
  comments documenting their own removal. Both now read the AST. This is R13's lesson,
  rediscovered in the gate that was supposed to enforce it.
