# `vllm-ascend` Fix-to-Test Recovery Mapping

Last updated: 2026-04-01

## 1. Goal

This document answers one question only:

> Which concrete code fix unblocked which test, or at least removed the first observed blocker?

To avoid mixing confirmed results with plausible-but-unverified correlations, the conclusions are split into two groups:

1. Confirmed recovery mappings: the related test case or parametrized subcase was re-run and passed.
2. Confirmed progress but not fully green: the original first failure disappeared, but downstream blockers still remain.

All dynamic re-runs in this round followed the same rule:

- single test case
- single idle NPU
- isolated process

## 1.1 Commit Scope for the Value Analysis

The fix-significance analysis in this document is tied to the actual `vllm-ascend` commit below:

- `d49937c9451cb898e804a4bad0e54a2ecc34531e`

This matters because both the fix-to-test mapping and the deeper engineering value must be grounded in the real diff, not inferred only from the observed test outcome.

---

## 2. Confirmed Recovery Mappings

| Changed file | Concrete code point | What changed | Tests confirmed to recover |
|---|---|---|---|
| `vllm_ascend/platform.py` | `NPUPlatform.check_and_update_config()` | Return early when `device_config.device_type != "npu"`; skip Ascend-specific config updates when `model_config is None`, so CPU and config-only paths are not polluted by the Ascend platform plugin. | `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py` (`1 passed`), `test_error_propagation.py` (`2 passed`), `test_invalid_blocks_correctness.py` (`3 passed`), `test_kv_load_failure_recovery.py` (`11 passed`), `test_decode_bench_connector.py` (`8 passed`), `test_offloading_connector.py` (`10 passed`), `test_remote_decode_lifecycle.py` (`4 passed`), `test_remote_prefill_lifecycle.py` (`6 passed`) |
| `vllm_ascend/ascend_config.py` | `AscendConfig.__init__()` model-related branches | Added explicit guards around accesses such as `model_config.is_deepseek_mla`, `hf_text_config`, and `enable_kv_nz`, preventing crashes in config-only flows. | `tests/v1/kv_connector/unit/test_config.py` (`6 passed`) |
| `vllm_ascend/platform.py` | `_get_npu_smi_hbm_capacity_mb()` and `get_device_total_memory()` | Implemented `NPUPlatform.get_device_total_memory()`, preferring `npu-smi` to query total HBM before falling back to `torch.npu` device properties. | `tests/v1/engine/test_engine_args.py` (`2 passed, 1 skipped`) |
| `vllm_ascend/platform.py` | `_ensure_ascend_worker_multiproc_method()` plus calls from `pre_register_and_update()` and `check_and_update_config()` | Defaulted Ascend to `spawn` when `VLLM_WORKER_MULTIPROC_METHOD` is not explicitly set, while preserving a user-specified `fork` or `spawn`. This avoids fork-unsafe `torch-npu` and ACL runtime behavior. | `tests/lora/test_qwenvl.py`: the previously stable first error `Invalid thread pool!` disappeared; we independently re-ran `test_qwen2vl_lora` and `test_qwen2vl_lora_beam_search` under the standard conditions above, and the user later re-ran the whole file and confirmed `6 passed`. |
| `vllm_ascend/lora/utils.py` | `Ascend*WithLoRA.can_replace_layer()` and `AscendMergedColumnParallelLinearVariableSliceWithLoRA` | Restored upstream-equivalent LoRA wrapper routing. `AscendMergedColumnParallelLinear` is no longer collapsed into one generic path; wrapper selection now depends on `packed_modules_list` length and, where needed, `output_sizes`. | The user verified that `tests/lora/test_add_lora.py` passed after this fix. It is also one of the prerequisites for stable recovery of `tests/lora/test_qwenvl.py`. |
| `vllm_ascend/lora/punica_npu.py` | `_get_token_lora_indices()`, `_shrink_decode()`, `_expand_decode()`, `_expand_slice_decode()` | The decode path no longer feeds the full `token_lora_indices` tensor into BGMV ops. Instead, it narrows the indices to the active `x.size(0)` window, fixing first-dimension mismatches among `x`, `y`, and the LoRA index tensor in multimodal LoRA decode. | `tests/lora/test_qwenvl.py`: the old LoRA decode dimension-mismatch first error disappeared; together with the default `spawn` fix, the user later confirmed `6 passed` for the whole file. |
| `vllm_ascend/worker/model_runner_v1.py` | `_torch_cuda_wrapper()` | After cleanup, the wrapper now re-raises the original exception instead of wrapping everything into `RuntimeError("NPUModelRunner init failed ...")`, preserving the exception type expected by upstream assertions. | `tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[...]` recovered for the 6 rejection subcases that were re-run independently on idle NPU 0-5. |

---

## 2.1 Why Each Fix Matters

This section answers a different question:

> What structural risk did each fix remove, what semantic boundary did it restore, and why does that matter for `vllm-ascend` as a platform plugin?

### 2.1.1 `check_and_update_config()` early return for non-NPU and no-model cases

The key value of this fix is not merely that several tests became quiet. Its real value is that it re-established the correct activation boundary of the Ascend platform plugin.

Before the fix, `NPUPlatform.check_and_update_config()` could still run Ascend-specific update logic on:

- CPU-only paths
- config-only paths where `VllmConfig` exists but `model_config` is still `None`

That behavior is structurally wrong for an out-of-tree platform plugin. A platform plugin should affect only the platform-specific execution paths that it truly owns. Once it starts modifying generic upstream configuration flows, it stops being a scoped adaptation layer and becomes a source of global side effects.

The real engineering value of the fix is:

1. It restores plugin scope.
   CPU-only and config-only flows now keep upstream semantics instead of being reinterpreted by Ascend-specific logic.
2. It reduces coupling to upstream evolution.
   Upstream `vLLM` legitimately uses `VllmConfig()` and `model_config=None` in many tests and helper paths. Without this guard, every new upstream config-only path is a potential regression trigger for the Ascend plugin.
3. It lowers install-time side effects.
   Installing `vllm-ascend` no longer implies that all config construction paths are exposed to NPU-specific mutation.

In short, this fix made `vllm-ascend` behave more like a disciplined platform plugin and less like a global config mutator.

### 2.1.2 Explicit `model_config` guards in `ascend_config.py`

This fix converted hidden assumptions into explicit contracts.

Before the fix, `AscendConfig.__init__()` accessed model-only fields such as:

- `model_config.is_deepseek_mla`
- `model_config.hf_text_config`
- feature gates behind `enable_kv_nz`

That implicitly assumed: “if AscendConfig is being initialized, then a valid model config must already exist.” Upstream `vLLM` does not guarantee that. `VllmConfig(model_config=None)` is legal.

The real value here is:

1. Failure semantics became interpretable.
   Instead of crashing at an arbitrary attribute access, the code now clearly separates:
   - optional paths that must no-op when no model exists
   - model-dependent features that must fail with an explicit requirement
2. Strong model assumptions are now contained.
   Constraints related to MLA, DeepSeek-specific branches, or KV-NZ are no longer allowed to leak into every Ascend config initialization flow.

This makes future Ascend-specific config work easier to extend safely because the code now distinguishes “safe without a model” from “requires model semantics” much more clearly.

### 2.1.3 `get_device_total_memory()` and the `npu-smi` first path

The value of this change is larger than “an API was added.” It moved total-memory probing away from a runtime-sensitive device initialization path.

The implementation now prefers:

- `npu-smi info -t memory -i <id> -c 0`

and only falls back to:

- `torch.npu.get_device_properties(device_id)`

Why this matters on Ascend:

- querying `torch.npu` properties is not always a harmless read
- in practice it may initialize parts of the runtime stack
- that is especially risky before worker creation, around multiprocessing boundaries, or in resource-probing paths

So the fix matters because it:

1. reduces early runtime initialization risk
2. makes memory probing a platform capability query instead of an execution-side effect
3. provides a more stable foundation for engine argument defaults, scheduling, and capacity planning

This is also valuable on its own because upstream `vLLM` already uses `current_platform.get_device_total_memory()` when deriving engine defaults. In other words, this is not just a test helper; it is part of the normal platform contract.

### 2.1.4 Defaulting Ascend workers to `spawn`

This fix promoted a runtime constraint from user folklore into platform policy.

Why Ascend is treated as non-fork-safe here:

- the `torch-npu` / ACL runtime stack is not modeled as fork-safe in this plugin
- if the parent process has already touched runtime state, `fork` can clone partially initialized thread pools, handles, and runtime state into children
- the observed failure pattern is consistent with that class of problem: child-side runtime initialization errors such as `Invalid thread pool!`

The code change is careful:

- if the user already set `VLLM_WORKER_MULTIPROC_METHOD`, the plugin respects it
- otherwise the plugin chooses the safer default, `spawn`

The deeper value is:

1. a known runtime hazard is encoded into the platform default
2. flaky, environment-sensitive failures are cut off at the entry point
3. users keep override control when they need it

This is why the fix is more than “making one test pass.” It encodes a platform safety rule into the plugin’s default behavior.

### 2.1.5 LoRA wrapper routing in `lora/utils.py`

This fix repaired a semantic collapse in the Ascend LoRA adaptation layer.

Upstream `vLLM` does not choose LoRA wrappers only by layer class name. It also distinguishes among:

- `packed_modules_list == 1`
- `packed_modules_list == 2`
- `packed_modules_list >= 3`
- and, in some cases, `output_sizes`

Before the fix, the Ascend plugin compressed too much of that routing logic into one coarse branch for `AscendMergedColumnParallelLinear`. That is dangerous because different wrapper classes expect different internal weight organizations and different `set_lora()` semantics.

The deeper value of the fix is:

1. it restores upstream-equivalent semantic routing
2. it prevents the plugin from flattening upstream’s type and packing distinctions
3. it improves compatibility structurally, not accidentally, for models that use packed, merged, or variable-slice linear layers

This is why the fix matters beyond one test file. It restores the plugin’s alignment with upstream LoRA layer semantics.

### 2.1.6 Removing the incorrect default overwrite of `output_sizes` in `ops/linear.py`

This change is small in diff size but important in design terms.

Previously, `AscendColumnParallelLinear.__init__()` would force:

- `output_sizes = [output_size]`

when `output_sizes` was originally `None`.

That is not a harmless fallback. It risks making later logic treat a plain column-parallel layer as if it carried packed or merged metadata. In other words, it papers over incorrect wrapper routing by manufacturing misleading structural metadata.

Removing that behavior matters because it:

1. preserves the semantic boundary between plain column-parallel layers and packed or merged layers
2. prevents a misleading workaround from becoming a permanent contract
3. forms a clean pair with the `lora/utils.py` routing fix

Together, those two changes restore the right answer in the right layer:

- routing logic picks the correct wrapper
- the base layer no longer pretends to have packed metadata that it does not truly own

### 2.1.7 Narrowing decode-time LoRA indices in `punica_npu.py`

This change is easier to understand if framed in one sentence:

During decode, the kernel should only see the LoRA indices for the tokens that are actually being processed right now.

Why the old behavior was wrong:

- `PunicaWrapperBase` stores `_token_lora_indices` in a max-sized buffer
- the active `x` passed into the decode kernel can be a smaller, narrowed view
- if decode still passes the full logical LoRA index tensor, then:
  - `x.shape[0]`
  - `y.shape[0]`
  - `token_lora_indices.shape[0]`
  may no longer match

The fix added `_get_token_lora_indices(x)` and narrowed the buffer to `x.size(0)` before calling:

- `bgmv_shrink`
- `bgmv_expand`
- `bgmv_expand_slice`

This is not unique to Ascend. Upstream XPU already does the same thing in [`punica_xpu.py`](/Users/francischen/Code2026/vllm/vllm/lora/punica_wrapper/punica_xpu.py), which strongly suggests that this is the correct dynamic-shape handling pattern for non-GPU Punica backends.

So the answer to “why do this?” is:

1. because the active decode window can be smaller than the preallocated metadata buffer
2. because the kernel needs shape-aligned per-token LoRA indices
3. because this matches an upstream non-GPU backend design, not an Ascend-only invention

So yes, this is best understood as an alignment fix toward upstream backend logic, especially the XPU Punica path.

### 2.1.8 Re-raising original exceptions in `_torch_cuda_wrapper()`

This fix restored transparent error semantics.

Before the change, the wrapper collapsed all failures into:

- `RuntimeError("NPUModelRunner init failed, error is ...")`

That is harmful because upstream tests and code paths may depend on the original exception type. Once everything becomes a generic `RuntimeError`, the plugin has effectively rewritten upstream’s failure contract.

The new behavior:

- logs the failure with `logger.exception(...)`
- re-raises the original exception

The value is straightforward:

1. the plugin improves observability without rewriting error meaning
2. debugging becomes more precise
3. rejection tests can assert the correct failure type again

This is exactly what a platform layer should do: add context, not destroy semantics.

---

## 3. Concrete Post-Fix Scenarios for `check_and_update_config()`

The value of the `check_and_update_config()` fix is easier to see through concrete scenarios.

### Case 1. CPU-only KV connector unit tests

Typical pattern:

- `VllmConfig(device_config=DeviceConfig("cpu"), ...)`
- the test is validating generic scheduler, KV transfer, or offloading behavior

Expected behavior after the fix:

- the Ascend platform plugin should not mutate the config at all
- no Ascend quantization detection
- no Ascend-specific compile/fusion adjustments
- no accidental model-dependent accesses

Representative recovered tests:

- `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py`
- `tests/v1/kv_connector/unit/test_offloading_connector.py`
- `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py`

### Case 2. Config-only construction with no model

Typical pattern:

- `VllmConfig(...)`
- `model_config is None`
- the test is only validating config defaults, helper utilities, or plumbing

Expected behavior after the fix:

- Ascend-specific updates must not run because there is no model semantic to consume
- config construction must remain valid

This is the exact class of issue protected both by:

- `platform.py` early return
- explicit `model_config` guards in `ascend_config.py`

### Case 3. Mixed environments where `vllm-ascend` is installed but the current path is not really an NPU execution path

Typical pattern:

- the plugin is importable in the environment
- but the current code path is not an actual Ascend execution path

Expected behavior after the fix:

- merely having the plugin installed must not change generic upstream behavior

This matters for CI, shared dev environments, and cross-platform regression testing.

---

## 4. Short FAQ

### Q1. Why is `vllm-ascend` treated as non-fork-safe?

Because the plugin explicitly encodes the assumption that `torch-npu` and the ACL runtime are unsafe defaults under `fork` once runtime state may already have been touched in the parent. The observed failure mode, such as `Invalid thread pool!`, is exactly the kind of post-fork runtime corruption this policy is designed to avoid.

This document uses “non-fork-safe” in that engineering sense:

- not “fork can never work”
- but “fork is not a safe platform default for this runtime stack”

### Q2. What is the standalone value of implementing `get_device_total_memory()`?

It is part of the normal platform contract in upstream `vLLM`, not just a test-only helper. Upstream engine argument handling already calls `current_platform.get_device_total_memory()` when computing memory-related defaults. So implementing it correctly in `vllm-ascend` improves:

- engine configuration correctness
- default memory estimation
- platform completeness
- startup stability when the implementation avoids premature runtime initialization

### Q3. Can we name concrete scenarios that are now valid after the `check_and_update_config()` fix?

Yes. Three concrete classes are now correctly handled:

1. CPU-only KV connector tests where Ascend should not interfere
2. config-only `VllmConfig` construction where `model_config=None`
3. shared environments where the Ascend plugin is installed but the current code path is not actually running on NPU

See Section 3 for representative examples.

### Q4. Why narrow LoRA indices in `punica_npu.py`? Is any other backend doing the same thing?

Yes. Upstream XPU already narrows `_token_lora_indices` to the active `x.size(0)` window before calling its decode-side BGMV ops. The Ascend fix follows the same pattern.

The reason is simple:

- the metadata buffer is preallocated
- the active decode tensor may be a smaller live slice
- the kernel should receive only the indices for the active slice

So this change is best viewed as an upstream-aligned backend fix, especially toward the existing XPU Punica design.

---

## 5. Short Takeaway

If we compress the whole mapping to the most important points, the result is:

1. `platform.py::check_and_update_config()` now stops Ascend from mutating CPU and no-model paths.
2. `ascend_config.py` now treats model-dependent assumptions as explicit contracts instead of hidden preconditions.
3. `platform.py::get_device_total_memory()` now fulfills a real upstream platform contract with a lower-side-effect memory probe path.
4. `platform.py` now defaults Ascend workers to `spawn`, which encodes a safer runtime policy.
5. `lora/utils.py` now restores upstream-equivalent LoRA wrapper routing.
6. `ops/linear.py` no longer fakes packed metadata on plain column-parallel layers.
7. `punica_npu.py` now aligns decode-time LoRA indices with the active token window, following the same pattern already used by upstream XPU.
8. `model_runner_v1.py::_torch_cuda_wrapper()` now preserves upstream exception semantics.
