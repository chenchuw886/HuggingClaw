# `vllm-ascend` 当前代码修复的影响范围分析

更新时间：2026-03-31

## 1. 目的

本文件回答的是另一个问题：

> 每个修改点到底影响了哪一层行为面，哪些测试/场景会直接受它影响，哪些则不应把责任继续外推给它。

这份文档与 `ASCEND_FIX_TO_TEST_MAPPING.md` 的区别是：

- 前者回答“哪个修复点对应哪些测试恢复”；
- 本文回答“这个修复点的运行时影响面边界在哪里”。

---

## 2. 总览表

| 修改点 | 直接影响面 | 已有证据 | 明确不覆盖的范围 |
|---|---|---|---|
| `vllm_ascend/platform.py::NPUPlatform.check_and_update_config()` 的 CPU/no-model 早返回 | 平台插件是否错误介入 CPU/config-only 路径 | `tests/v1/kv_connector/unit/*` 一批 upstream 纯逻辑单测恢复通过 | 不改变真实 NPU 执行语义；不修复 Ascend 算子/运行时错误 |
| `vllm_ascend/ascend_config.py::AscendConfig.__init__()` 的 `model_config` 显式保护 | 纯配置初始化路径是否能在 `model_config is None` 时安全返回/拒绝 | `tests/v1/kv_connector/unit/test_config.py` 通过 | 不解决真实模型加载、推理、图编译问题 |
| `vllm_ascend/platform.py::_get_npu_smi_hbm_capacity_mb()` + `get_device_total_memory()` | 设备总显存查询语义，以及查询阶段是否提前触发 `torch.npu` 初始化 | `tests/v1/engine/test_engine_args.py` 通过；`tests/ut/test_platform.py` 覆盖 `npu-smi` 优先分支 | 不提高可用空闲显存；不解决运行中 OOM/显存阈值不足 |
| `vllm_ascend/platform.py::_ensure_ascend_worker_multiproc_method()` | 库模式/worker 进程模型默认值；Ascend 下 `fork` 带来的 ACL runtime 不安全问题 | `tests/lora/test_qwenvl.py` 旧的 `Invalid thread pool!` 首错消失；整文件已手动复跑 `6 passed` | 不替代用户显式设置；不解决与进程模型无关的 LoRA 形状错误或真实显存不足 |
| `vllm_ascend/lora/punica_npu.py::_get_token_lora_indices()` 及 decode 路径调用点 | 多模态 LoRA decode 阶段 `x/y/token_lora_indices` 第一维对齐关系 | `tests/lora/test_qwenvl.py` 旧的 LoRA decode 维度错配首错消失；整文件已手动复跑 `6 passed` | 不改变 prefill 路径本身；不解决 worker 启动、ACL runtime fork-unsafe、资源竞争 |
| `vllm_ascend/worker/model_runner_v1.py::NPUModelRunner.__init__()` 中 `mm_registry` / `supports_mm_inputs` / `mm_budget` | 多模态 runner 生命周期状态是否与 upstream 对齐 | `tests/models/multimodal/generation/test_granite_speech.py` 已越过 `mm_budget` 缺失首错 | 不保证多模态业务输出正确；不解决后续 logprobs 对齐或模型本身支持性 |
| `vllm_ascend/worker/model_runner_v1.py::NPUModelRunner.__init__()` 中 `custom_logitsprocs` / `logitsprocs_need_output_token_ids` | custom logits processor 离线路径的 `InputBatch` 初始化契约 | 代码面已与 upstream 初始化语义对齐；与 `test_custom_offline.py` 路径强相关 | 当前未做单改动 A/B 隔离，不单独宣称某个上游用例完全由它修复 |
| `vllm_ascend/worker/model_runner_v1.py::_torch_cuda_wrapper()` | `GPUModelRunner` 初始化失败时暴露给上层的异常类型 | `tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[...]` 6 个 rejection 子例通过 | 不改变底层真实异常原因；只改变上层能否看到并断言原始异常类型 |

---

## 3. 分修改点说明

### 3.1 `platform.py::check_and_update_config()`

- **影响对象**
  - 所有会经过平台插件，但实际目标设备不是 `npu` 的配置路径。
  - 所有只验证 config/scheduler/connector 纯逻辑、不需要真实 NPU 模型配置的 upstream 单测。
- **为什么它重要**
  - 修复前，Ascend 平台插件会在 CPU/config-only 测试里继续下钻到 Ascend 专属配置，制造额外噪音。
  - 修复后，这些测试重新回到 upstream 原始语义。
- **边界**
  - 这是“去噪”型修复，不是“新增 Ascend 功能”型修复。
  - 因此它恢复通过的 `kv_connector` 多数测试，不应被解读成 Ascend 特有 connector 语义被新增实现。

### 3.2 `ascend_config.py::AscendConfig.__init__()`

- **影响对象**
  - `model_config` 为空或不完整时仍会构造 `AscendConfig` 的路径。
  - `enable_kv_nz` / `hf_text_config` / `is_deepseek_mla` 等依赖模型元数据的配置分支。
- **为什么它重要**
  - 它把“纯配置场景”和“真实模型场景”分开处理，避免空对象访问把测试提前打断。
- **边界**
  - 一旦进入真实模型推理，该修复不参与算子、图编译或精度行为。

### 3.3 `platform.py::get_device_total_memory()`

- **影响对象**
  - 依赖 `current_platform.get_device_total_memory()` 的参数校验和引擎前置判断。
  - 在 worker 派生前就需要读取总显存的路径。
- **为什么它重要**
  - 先查 `npu-smi` 可以避免过早触发 `torch.npu` 初始化，从而减小与多进程初始化顺序相关的副作用。
- **边界**
  - 它读取的是**总显存**而非**空闲显存**；因此不能把运行期的“可用显存不足”问题继续归因给它。

### 3.4 `platform.py::_ensure_ascend_worker_multiproc_method()`

- **影响对象**
  - 所有未显式设置 `VLLM_WORKER_MULTIPROC_METHOD` 的 Ascend 库模式入口。
  - 所有会在 worker 子进程里再次接触 ACL runtime / `torch-npu` 初始化的路径。
- **为什么它重要**
  - Ascend 运行时对 `fork` 不安全；默认改成 `spawn` 后，旧的 `Invalid thread pool!` 首错不再出现。
- **边界**
  - 如果用户显式设置了环境变量，本修复不会覆盖用户选择。
  - 如果后续失败是 LoRA shape、模型能力缺口、显存不足、算子报错，本修复不应再背锅。

### 3.5 `punica_npu.py` 的 token-index 收窄

- **影响对象**
  - 多模态 LoRA decode 阶段对 `token_lora_indices` 的使用。
  - `x.size(0)` 小于缓存长度时，需要按当前 token 数收窄索引的路径。
- **为什么它重要**
  - 它恢复了 upstream decode 阶段隐含的维度对齐契约，是 `test_qwenvl.py` 能继续向后运行的直接前提。
- **边界**
  - 它只修 LoRA decode 维度对齐，不处理进程模型、worker 初始化或资源竞争。

### 3.6 `model_runner_v1.py` 的 `mm_budget` 状态补齐

- **影响对象**
  - 所有使用 `NPUModelRunner`、且上游期望 runner 持有 `mm_budget` 生命周期状态的多模态路径。
- **为什么它重要**
  - 它把 Ascend runner 拉回到 upstream 的对象状态契约，避免在真正业务逻辑之前就因属性缺失崩溃。
- **边界**
  - 它只是“把程序带到下一层失败点”，不等同于多模态行为已经对齐。

### 3.7 `model_runner_v1.py` 的 custom logits processor 初始化补齐

- **影响对象**
  - 通过 `LLM(..., logits_processors=...)` 传入自定义 logits processor 的离线路径。
  - 需要 `InputBatch` 正确知道“是否需要 output token ids”的路径。
- **为什么它重要**
  - 这是一次和 upstream 初始化语义对齐的修复，否则 Ascend 路径即便没有异常类型问题，也可能在后续 processor 装配/采样元数据上继续偏离。
- **边界**
  - 当前没有做单独 A/B 证据隔离，所以不宜把所有 `test_custom_offline.py` 通过都单独归功给它。

### 3.8 `_torch_cuda_wrapper()` 保留原始异常类型

- **影响对象**
  - 所有需要在 `GPUModelRunner` 初始化失败时，依赖原始异常类型/消息做断言的上游测试。
- **为什么它重要**
  - 修复前，Ascend 统一包成 `RuntimeError`，会破坏 upstream 对 `ValueError` 等异常类型的断言。
  - 修复后，测试终于能看到真正的拒绝理由。
- **边界**
  - 它不让错误“消失”，只是让错误以正确的类型暴露出来。

---

## 4. 当前建议的阅读顺序

如果后续还要继续追某个失败，建议按下面顺序判断它属于哪一层：

1. **先看是不是 CPU/config-only 噪音**
   - 优先排查 `check_and_update_config()` / `AscendConfig.__init__()` 影响面。
2. **再看是不是设备查询或多进程初始化问题**
   - 优先排查 `get_device_total_memory()` 与默认 `spawn`。
3. **再看是不是 LoRA / 多模态运行期契约问题**
   - 优先排查 `punica_npu.py`、`mm_budget`、custom logits processor 初始化。
4. **最后看是不是异常暴露层问题**
   - 如果真实错误被包坏，再回看 `_torch_cuda_wrapper()`。

---

## 5. 最短结论

- `platform.py` 里的三个修复点分别解决了：CPU/config-only 污染、总显存查询契约、以及 Ascend 下默认进程模型不安全。
- `punica_npu.py` 解决的是 LoRA decode 维度契约，不是进程模型问题。
- `model_runner_v1.py` 目前同时承载了三类影响面：多模态生命周期状态、custom logits processor 初始化语义、以及原始异常类型保留。
- 后续写 FIX_TO_TEST 映射时，应该只对已经拿到稳定补跑证据的改动宣称“直接修复”，其余则像本文一样先标清影响面边界。