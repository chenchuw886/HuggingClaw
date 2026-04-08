# Beta test files failure analysis (in progress)

## Environment

- vllm commit: `b31e9326a7d9394aab8c767f8ebe225c65594b60`
- vllm-ascend commit: `e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45`
- Python: `3.11.14`
- torch: `2.9.0+cpu`
- torch-npu: `2.9.0`
- npu-smi: `25.5.1`
- Hardware used for serial runs: single-card exclusive runs on `ASCEND_RT_VISIBLE_DEVICES=7` and later `6`
- Extra env/workarounds: `HF_ENDPOINT=https://hf-mirror.com` first; `VLLM_USE_MODELSCOPE` only as fallback when HF mirror is unavailable

## Current results

| Test file | Status | True root cause | Ownership | vAscend code fix can pass? | Notes |
| --- | --- | --- | --- | --- | --- |
| `tests/config/test_config_generation.py` | failed | `vllm_ascend/platform.py` mutates compilation-related config (`enable_npugraph_ex`) depending on empty vs unset `CUDA_VISIBLE_DEVICES`, breaking upstream config equivalence contract. | vllm-ascend | Yes | Not a network issue; reproducible after model config resolution. |
| `tests/engine/test_arg_utils.py` | failed | `vllm_ascend/platform.py` resets user-selected `attention_config.backend` to `None`, violating upstream CLI/config propagation contract. | vllm-ascend | Yes | Behavior contract regression at plugin adaptation boundary. |
| `tests/entrypoints/openai/test_default_mm_loras.py` | failed | Clean single-card rerun on an idle NPU still OOMs during engine init/profile for Phi-4 multimodal + default MM LoRA (`Tried to allocate 20 GiB`, only ~18.9 GiB free on the visible card). | vllm-ascend / runtime capacity | Maybe, but not a quick functional fix | Confirmed not caused by competing NPU processes; this is a real single-card memory blow-up. |
| `tests/entrypoints/openai/test_return_tokens_as_ids.py` | failed | Request path reaches prompt-logprobs + LoRA logits computation, then `vllm_ascend/lora/lora_ops.py::bgmv_shrink` fails with `the first dimension of x, y, indices should be same`. | vllm-ascend | Yes | Clear Ascend LoRA op shape-handling bug on completion API path. |
| `tests/entrypoints/openai/test_video.py` | failed | First multimodal video chat request succeeds, but second-turn chat request with history returns server-side `500 InternalServerError` on Ascend; log does not expose the inner stack. | Unknown yet (`vllm` or `vllm-ascend`) | Unclear | Looks like a multi-turn multimodal state/cache path issue, not a network/download blocker. Needs one more debug rerun for exact ownership. |
| `tests/lora/test_deepseekv2_tp.py` | failed | Clean single-card rerun on an idle NPU still reaches model init and OOMs in `vllm_ascend/ops/fused_moe/fused_moe.py` while creating MoE weights (`Tried to allocate 706 MiB`, only ~72 MiB free). | vllm-ascend / runtime capacity | Maybe, but not a quick functional fix | Confirmed not caused by other NPU processes. |
| `tests/lora/test_qwen3moe_tp.py` | failed | Clean single-card rerun on an idle NPU still reaches model init and OOMs in `vllm_ascend/ops/fused_moe/fused_moe.py` while creating MoE weights (`Tried to allocate 770 MiB`, only ~546 MiB free). | vllm-ascend / runtime capacity | Maybe, but not a quick functional fix | Confirmed not caused by other NPU processes. |
| `tests/model_executor/test_model_load_with_params.py` | passed | Passes cleanly under HF-only rerun (`3 passed`). Earlier failure was pure ModelScope noise. | environment / mirror selection | N/A | Do not treat the earlier main-batch failure as a real blocker. |
| `tests/models/language/generation/test_hybrid.py` | failed | HF-only rerun reaches real Ascend engine init for Mamba, then `vllm_ascend/worker/model_runner_v1.py::_allocate_kv_cache_tensors` asserts `Some layers are not correctly initialized`; KV-cache layer-name allocation does not match hybrid/Mamba layer set. | vllm-ascend | Yes | Real adaptation bug after removing mirror noise. |
| `tests/models/multimodal/generation/test_whisper.py` | failed | HF-only rerun reaches real config validation and fails because Ascend platform re-enables ACL graph defaults even after `enforce_eager=True`, producing a `VllmConfig` validation error (`mode=none` but platform forces `PIECEWISE`/ACL graph expectations). | vllm-ascend | Yes | Another config contract regression at the Ascend platform layer. |
| `tests/models/multimodal/pooling/test_siglip.py` | failed | Image pooling path crashes in `vllm_ascend/patch/worker/patch_multimodal_merge.py` with bogus placeholder accounting (`Attempted to assign 1 = 1 multimodal tokens to 4429202204900686416 placeholders`). | vllm-ascend | Yes | Clear multimodal merge adaptation bug on Ascend. |
| `tests/plugins_tests/test_io_processor_plugins.py` | failed | After installing `imagehash` and `timm`, rerun progresses further but still fails during model inspection because optional package `terratorch` is missing for the `Terratorch` architecture. | environment / dependency | No code fix needed in vllm-ascend | Still dependency-gated; not yet a real Ascend runtime failure. |
| `tests/plugins_tests/test_stats_logger_plugins.py` | failed | After installing the dummy plugin package, plugin discovery succeeds, but engine startup then fails in Ascend init: `MemorySnapshot()` asserts because the platform lacks a usable `current_device` memory hook. | vllm-ascend / platform integration | Yes | The original collection/setup failure was only noise; current failure is a real Ascend integration issue. |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | failed | Parser returns reasoning text for plain output without `<think>` tags; failure is in upstream reasoning parser behavior, not Ascend runtime. | vllm | No | Pure parser contract test, not Ascend-sensitive. |
| `tests/test_seed_behavior.py` | failed | Test calls `Platform.seed_everything`, but `vllm.platforms.interface.Platform` has no such API. | vllm / test | No | Not an Ascend adaptation issue. |
| `tests/v1/ec_connector/unit/test_ec_example_connector.py` | failed | Upstream example connector implementation lacks `has_caches`, while the unit test expects it. | vllm | No | Example connector API drift, not Ascend-specific. |
| `tests/v1/entrypoints/openai/serving_responses/test_function_call.py` | passed | Passed in the initial serial batch. | passed | N/A | No Ascend blocker observed. |
| `tests/v1/worker/test_utils.py` | failed | `vllm/v1/worker/utils.py::bind_kv_cache` raises `NotImplementedError` on Ascend when multiple attention layers map to the same layer index (draft model case), while CUDA/XPU/CPU explicitly tolerate this test path. | vllm / platform gap | Maybe | Not clearly isolated to `vllm-ascend` code, but it is an Ascend execution gap that likely needs platform-specific handling somewhere in the adaptation boundary. |
| `tests/v1/streaming_input/test_scheduler_streaming.py` | failed | Test helper constructs `VllmConfig` with a `MagicMock` model config; scheduler init then trips the upstream encoder-decoder multimodal assertion. This looks like a brittle upstream/mock-contract issue rather than an Ascend runtime failure. | vllm / test | No | Failure happens before any real Ascend-specific execution path. |
| `tests/compile/correctness_e2e/test_sequence_parallel.py` | failed | Upstream compile test assumes CUDA-style device capability tuples; Ascend `current_platform.get_device_capability()` returns `None`, so the FP8 skip guard crashes with `TypeError`. | vllm-ascend / platform interface | Yes | Minimal fix is to return a comparable capability sentinel or make the platform/test guard robust. |
| `tests/compile/test_startup.py` | failed | Original run tripped the free-memory threshold, but a clean single-card rerun on an idle NPU no longer failed on memory; it advanced and then failed functionally with `TypeError: phimoe_routing_function() got an unexpected keyword argument 'global_num_experts'`. | vllm-ascend / functional bug | Yes | The earlier memory-threshold signal was likely secondary/noisy; current actionable blocker is functional. |
| `tests/distributed/test_elastic_ep.py` | passed | Passed in the HF-only continuation batch. | passed | N/A | No Ascend blocker observed. |
| `tests/entrypoints/openai/test_realtime_validation.py` | failed | HF-only log reaches real Voxtral engine init, then `vllm/model_executor/models/whisper_causal.py` raises `NotImplementedError` because Whisper block-pooling attention does not support `AscendAttentionBackend`. | vllm-ascend | Yes | Real backend coverage gap for Voxtral realtime / Whisper block-pooling path. |
| `tests/models/multimodal/generation/test_voxtral_realtime.py` | failed | Same Voxtral/Whisper backend gap as `test_realtime_validation`: model init reaches Whisper block-pooling attention creation and fails because `AscendAttentionBackend` is unsupported there. | vllm-ascend | Yes | Same fix cluster as realtime validation. |
| `tests/models/multimodal/pooling/test_colmodernvbert.py` | failed | Model/config loading fails before Ascend runtime because installed `transformers` does not recognize the `modernvbert` architecture. | environment / runtime | No | Dependency/version support gap, not an Ascend adaptation bug. |
| `tests/renderers/test_sparse_tensor_validation.py` | failed | Malicious sparse tensor validation path raises an unexpected `_pickle.UnpicklingError` from the Ascend runtime stack instead of the upstream-expected `RuntimeError`/`ValueError`. | runtime / vllm-ascend integration | Yes | Likely fixable by normalizing or wrapping Ascend-side deserialization exceptions to match upstream contracts. |
| `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py` | failed | Integration path errors with `ValueError: Unknown speculative decoding method: extract_hidden_states`. | vllm-ascend | Yes | Looks like an adaptation/config registration gap for the extract-hidden-states path. |

## Execution note

- I initially enabled `VLLM_USE_MODELSCOPE=True` too early for the main batch. This does **not** follow `SKILL.md` rule 3, which says ModelScope is only a fallback when HF mirror is unavailable.
- I corrected the workflow: ongoing reruns now use `HF_ENDPOINT=https://hf-mirror.com` first and keep `VLLM_USE_MODELSCOPE` unset unless HF mirror itself proves unavailable.

## Raw run notes

- Each file is executed serially with `python -m pytest <file> -x -q`.
- Per-file logs are stored under `HuggingClaw/test_failure_analysis_guideline/beta_run_logs/`.
- The batch is still running; this report will be updated incrementally.
