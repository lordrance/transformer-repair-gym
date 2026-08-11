# TARGET_POLICY_CONFIG

The frozen sampling configuration for Phase 0.6 target-policy calibration.

**Everything below was verified by introspecting the installed SDK
(`tinker==0.25.0`, `tinker-cookbook==0.5.3`) and the live Tinker docs on
2026-08-10 — not copied from an earlier report.** The two doc URLs used in
Phase 0 (`/tinker/sampling/`, `/api-reference/samplingclient`) now 404, so
signatures come from the package itself.

## Pinned configuration

| field | value | source |
|---|---|---|
| **model id** | `Qwen/Qwen3.5-4B` | listed as available on the Tinker models page |
| **renderer** | **`qwen3_5`** | `model_info.get_recommended_renderer_name("Qwen/Qwen3.5-4B")` |
| **thinking mode** | **ON** | `qwen3_5` opens `<think>\n`; the alternative `qwen3_5_disable_thinking` closes it immediately |
| temperature | `1.0` | matches the DeepSeek baseline; also what a GRPO rollout would use |
| top_p / top_k | `1.0` / `-1` | SDK defaults, left untouched |
| max_tokens | `16384` | headroom, see *Truncation* below |
| seed | `None` (unset) | 4 rollouts must be independent samples |
| stop | `[248046]` | `renderer.get_stop_sequences()` |
| context window | 65,536 | Tinker models page |
| num_samples per prompt | 4 | one `sample()` call per task |

### Why thinking is ON

`get_recommended_renderer_names` returns `('qwen3_5', 'qwen3_5_disable_thinking')`,
in that order. Thinking-on is the default recommendation, it is what the
DeepSeek baseline effectively used (90–97 % of its completion tokens were
reasoning), and it is what an RL run on a reasoning task would train.

**This choice is now frozen.** If the Qwen baseline comes back weak, the correct
response is to redesign the tasks or build a curriculum — *not* to switch the
renderer and re-run. Changing the template to improve a baseline number makes
the number meaningless, and silently decouples the calibration from the training
config it exists to predict.

Both renderers were measured anyway, because the difference is 2 tokens per
prompt and knowing it costs nothing: see `artifacts/qwen_preflight.json`.

## Verified API surface (`tinker==0.25.0`)

```python
service = tinker.ServiceClient()                       # reads TINKER_API_KEY from env
client  = service.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
tok     = client.get_tokenizer()

renderer = renderers.get_renderer("qwen3_5", tok, model_name="Qwen/Qwen3.5-4B")
prompt   = renderer.build_generation_prompt(messages)   # -> tinker.ModelInput
params   = tinker.SamplingParams(max_tokens=16384, temperature=1.0,
                                 stop=renderer.get_stop_sequences())

response = client.sample(prompt=prompt, num_samples=4,
                         sampling_params=params).result()
```

Exact signatures, as introspected:

```
ServiceClient.create_sampling_client(model_path=None, base_model=None,
                                     retry_config=None) -> SamplingClient
SamplingClient.sample(prompt: ModelInput, num_samples: int,
                      sampling_params: SamplingParams,
                      include_prompt_logprobs: bool = False,
                      topk_prompt_logprobs: int = 0)
                      -> ConcurrentFuture[SampleResponse]
SamplingParams(max_tokens=None, seed=None, stop=None,
               temperature=1, top_k=-1, top_p=1)
SampleResponse(sequences: Sequence[SampledSequence], ...,
               prompt_cache_hit_tokens: int = 0)
SampledSequence(stop_reason: StopReason, tokens_np, logprobs_np, ...)
Renderer.build_generation_prompt(messages, role='assistant', prefill=None) -> ModelInput
Renderer.get_stop_sequences() -> list[str] | list[int]
Renderer.parse_response(response: list[int]) -> tuple[Message, ParseTermination]
```

There is **no `usage` object**. Token accounting is assembled from
`ModelInput.length` (prefill), `len(SampledSequence.tokens)` (sampled) and
`SampleResponse.prompt_cache_hit_tokens`. The runner does exactly that.

**Renderers live in `tinker_cookbook`, not `tinker`.** `tinker` alone cannot
build a chat prompt. Both packages are pinned in the venv.

## Pricing snapshot

**Snapshot date: 2026-08-10.** Source: Tinker models & pricing page. Re-check
before relying on it; Phase 0 already shipped one stale price table.

| item | `Qwen3.5-4B`, USD / 1M tokens |
|---|---|
| prefill | **$0.33** |
| cached prefill (80 % discount) | **$0.066** |
| sample | **$1.005** |
| training | $0.737 *(not used this phase)* |

## Measured prompt sizes (pinned renderer, no key required)

Rendered through `qwen3_5` with the real HF tokenizer:

| task | prompt tokens | headroom in 64K |
|---|---|---|
| `t1_causal_mask_off_by_one` | 3,110 | 62,426 |
| `t2_rope_pairing_convention` | 3,089 | 62,447 |
| `t3_rmsnorm_missing_upcast` | 3,034 | 62,502 |
| `t4_grad_accum_normalization` | 958 | 64,578 |
| `t5_loss_ignore_index_dropped` | 3,108 | 62,428 |
| **total (5 prompts)** | **13,299** | |

Prefill for the whole run is billed once per prompt (4 samples share it):
13,299 tokens ≈ **$0.0044**. Sampling dominates:

| mean sampled / rollout | total for 20 rollouts | per rollout |
|---|---|---|
| 1,000 | $0.0245 | $0.00122 |
| 3,000 | $0.0647 | $0.00323 |
| 6,000 | $0.1250 | $0.00625 |
| 12,000 | $0.2456 | $0.01228 |

DeepSeek v4-pro averaged 2,151 completion tokens on these same prompts. A 4B
model with thinking on will plausibly use more. **Expected cost of the
calibration: a few cents to ~$0.25.**

## Truncation

`max_tokens = 16384`, well above any plausible need, because Phase 0.5 lost 20
trajectories to exactly this: at `max_tokens=8000`, `deepseek-v4-flash` spent
99 % of its budget reasoning and never emitted a diff. Every "failure" was
`finish_reason == "length"` — a configuration artifact that would have been
reported as a difficulty finding.

The runner records `stop_reason` and a `truncated` flag per rollout and prints a
warning if any sample hits the cap. **Any truncated rollout is a config failure
and must be excluded from difficulty conclusions.**

## What is held byte-identical to the DeepSeek baseline

So that any difference is attributable to the model and not the task:

- the five task specs, their mutations, symptoms and workspaces
- `SYSTEM_PROMPT` and the user-message construction in `build_dataset()`
- the unified-diff protocol, parser, fuzzy applier and INVALID definition
- naive verifier v2 and the hardened verifier, unchanged
- the independent semantic equivalence probe (`PROBE_SEED = 987654321`)

## Enforced prohibitions

`scripts/run_qwen_baseline.py` creates exactly one Tinker object: a base-model
`SamplingClient`. It imports no `TrainingClient`, calls no `forward_backward`,
`optim_step` or `save_weights`, and writes no checkpoint. No task generation, no
LoRA, no GRPO.

## Credential handling

`TINKER_API_KEY` is read from the environment by `tinker.ServiceClient()` and by
nothing else. It is not written to source, config, JSON, JSONL, logs or the
README, and it is not printed — not even a prefix. The runner exits with a usage
message if the variable is absent.
