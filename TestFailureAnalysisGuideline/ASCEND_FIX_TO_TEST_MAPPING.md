# `vllm-ascend` 具体改动点 -> 用例恢复映射表

更新时间：2026-03-31

## 1. 目的

本文件只回答一个问题：

> 哪一个具体代码改动点，对应让哪些用例跑通，或者至少越过了之前的首个阻塞点。

为避免“相关但未证实”的映射混入，本文把结论分成两类：

1. **已明确跑通映射**：改动后，相关用例/参数化子例已实际补跑通过；
2. **已明确推进但未全绿**：改动后，旧首错消失，但文件级别/全量用例还存在下游阻塞。

本轮动态补跑统一要求：**单用例、单空闲 NPU、独立进程**。

---

## 2. 已明确跑通映射

| 改动文件 | 具体代码点 | 改动内容 | 改后跑通的用例 |
|---|---|---|---|
| `vllm_ascend/platform.py` | `NPUPlatform.check_and_update_config()` | 当 `device_config.device_type != "npu"` 时直接早返回；`model_config is None` 时跳过 Ascend 专属配置更新，避免 CPU/config-only 场景被平台插件污染。 | `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py` (`1 passed`)、`test_error_propagation.py` (`2 passed`)、`test_invalid_blocks_correctness.py` (`3 passed`)、`test_kv_load_failure_recovery.py` (`11 passed`)、`test_decode_bench_connector.py` (`8 passed`)、`test_offloading_connector.py` (`10 passed`)、`test_remote_decode_lifecycle.py` (`4 passed`)、`test_remote_prefill_lifecycle.py` (`6 passed`) |
| `vllm_ascend/ascend_config.py` | `AscendConfig.__init__()` 中 `model_config` 相关分支 | 对 `model_config.is_deepseek_mla`、`hf_text_config`、`enable_kv_nz` 等访问增加显式保护，避免纯配置场景崩溃。 | `tests/v1/kv_connector/unit/test_config.py` (`6 passed`) |
| `vllm_ascend/platform.py` | `_get_npu_smi_hbm_capacity_mb()` + `get_device_total_memory()` | 补齐 `NPUPlatform.get_device_total_memory()`，并优先走 `npu-smi` 获取总显存，避免在 fork 前提前初始化 `torch.npu`。 | `tests/v1/engine/test_engine_args.py` (`2 passed, 1 skipped`) |
| `vllm_ascend/platform.py` | `_ensure_ascend_worker_multiproc_method()` + `pre_register_and_update()` / `check_and_update_config()` | 当用户未显式设置 `VLLM_WORKER_MULTIPROC_METHOD` 时，Ascend 默认改用 `spawn`，但仍保留用户手动指定的 `fork/spawn`。该改动用于规避 `torch-npu` / ACL runtime 的 fork-unsafe 行为。 | `tests/lora/test_qwenvl.py`：先前稳定出现的 `Invalid thread pool!` 首错已消失；我们已在“单用例、单空闲 NPU、独立进程”条件下独立补跑通过 `test_qwen2vl_lora`、`test_qwen2vl_lora_beam_search`，且用户随后按同一口径手动复跑整文件确认 `6 passed`。 |
| `vllm_ascend/lora/punica_npu.py` | `_get_token_lora_indices()`、`_shrink_decode()`、`_expand_decode()`、`_expand_slice_decode()` | decode 路径不再直接使用完整 `token_lora_indices`，而是按当前 `x.size(0)` 收窄，修复多模态 LoRA 场景下 `x/y/indices` 第一维不一致。 | `tests/lora/test_qwenvl.py`：旧的 LoRA decode 维度错配首错已被消除；结合上面的默认 `spawn` 修复后，用户已按“单用例、单空闲 NPU、独立进程”口径手动复跑整文件确认 `6 passed`。 |
| `vllm_ascend/worker/model_runner_v1.py` | `_torch_cuda_wrapper()` | 异常清理后改为 `raise` 原始异常，而不是统一包装成 `RuntimeError("NPUModelRunner init failed ...")`，从而保留 upstream 断言所需的异常类型。 | `tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT-pooling]`、`[...ENTRYPOINT-spec_dec]`、`[...FQCN-pooling]`、`[...FQCN-spec_dec]`、`[...CLASS-pooling]`、`[...CLASS-spec_dec]` 已在空闲 NPU 0-5 上逐例独立补跑通过。 |

---

## 3. 已明确推进但未全绿

| 改动文件 | 具体代码点 | 改动内容 | 推进效果 | 当前剩余阻塞 |
|---|---|---|---|---|
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.__init__()` 中 `mm_registry` / `supports_mm_inputs` / `mm_budget` | 补齐 upstream 多模态 budget 状态，修复 `NPUModelRunner` 缺少 `mm_budget` 的生命周期状态缺口。 | `tests/models/multimodal/generation/test_granite_speech.py` 已越过最初的 `AttributeError: 'NPUModelRunner' object has no attribute 'mm_budget'`。 | 报错为 logprobs 与 HF 结果对不齐，Granite 这类音频+lora 的模型本身 vllm-ascend 暂未明确支持，故不作进一步深入分析了。 |

---

## 4. 对 kv_connector 的特别说明

`tests/v1/kv_connector/unit/` 这一组里，要区分两种“跑通”：

### 4.1 直接修复跑通
只有：

- `tests/v1/kv_connector/unit/test_config.py`

它直接依赖：

- `vllm_ascend/platform.py`
- `vllm_ascend/ascend_config.py`

### 4.2 去噪后恢复通过

以下文件现在虽然都通过了，但更准确地说是：

- Ascend 不再错误介入 CPU/config-only 路径后，
- upstream 的 scheduler / connector 逻辑恢复为原始语义并通过。

对应文件：

- `test_cache_pollution_prevention.py`
- `test_error_propagation.py`
- `test_invalid_blocks_correctness.py`
- `test_kv_load_failure_recovery.py`
- `test_decode_bench_connector.py`
- `test_offloading_connector.py`
- `test_remote_decode_lifecycle.py`
- `test_remote_prefill_lifecycle.py`

因此这些文件不应解读成：

- “Ascend 新实现了某个 connector 语义”

更准确的解读是：

- “Ascend 去掉了对 upstream CPU-only 测试的错误干扰，因此这些 upstream 逻辑测试恢复通过。”

---

## 5. 最短结论

如果只保留最关键的改动 -> 用例映射，可以记成下面 7 条：

1. `platform.py::check_and_update_config()` CPU/no-model 早返回
   - 直接去噪并恢复一批 `tests/v1/kv_connector/unit/*` upstream 逻辑测试。
2. `ascend_config.py` 的 `model_config` 显式保护
   - 直接修复 `tests/v1/kv_connector/unit/test_config.py`。
3. `platform.py::get_device_total_memory()` + `npu-smi` 路径
   - 直接修复 `tests/v1/engine/test_engine_args.py`。
4. `platform.py` 默认未指定时改用 `spawn`
   - 直接消除了 `test_qwenvl.py` 中由 `fork` 引起的 `Invalid thread pool!` 首错；当前已与用户手动整文件复跑结果共同收敛为 `6 passed`。
5. `punica_npu.py` 的 token-index 收窄
   - 直接消除了 `tests/lora/test_qwenvl.py` 旧的 LoRA decode 维度错配；与默认 `spawn` 修复共同构成当前整文件 `6 passed` 的代码前提。
6. `model_runner_v1.py::_torch_cuda_wrapper()` 保留原始异常类型
   - 直接修复 `test_custom_offline.py::test_rejects_custom_logitsprocs` 的 6 个 rejection 子例。
7. `model_runner_v1.py` 的 `custom_logitsprocs` / `logitsprocs_need_output_token_ids` 初始化补齐
   - 已确认影响 custom logits processor 初始化路径，但当前尚未做单改动 A/B 隔离，因此先纳入影响范围分析，不在本表中单独宣称某个 upstream 用例由它独立修复。
