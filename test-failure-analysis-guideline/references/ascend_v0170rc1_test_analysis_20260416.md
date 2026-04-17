# vLLM v0.17.0rc1 上游用例在 Ascend NPU 上的批量分析

更新时间：2026-04-16

## 1. 分析范围

目标清单来自：

- `test_files_need_analysis_v0170rc1.md`

共 20 个测试文件：

1. `tests/config/test_config_generation.py`
2. `tests/engine/test_arg_utils.py`
3. `tests/entrypoints/openai/test_default_mm_loras.py`
4. `tests/entrypoints/openai/test_return_tokens_as_ids.py`
5. `tests/entrypoints/openai/test_video.py`
6. `tests/lora/test_deepseekv2_tp.py`
7. `tests/lora/test_qwen3moe_tp.py`
8. `tests/model_executor/test_model_load_with_params.py`
9. `tests/models/language/generation/test_hybrid.py`
10. `tests/models/multimodal/generation/test_whisper.py`
11. `tests/plugins_tests/test_io_processor_plugins.py`
12. `tests/v1/ec_connector/unit/test_ec_example_connector.py`
13. `tests/v1/streaming_input/test_scheduler_streaming.py`
14. `tests/compile/correctness_e2e/test_sequence_parallel.py`
15. `tests/compile/test_startup.py`
16. `tests/distributed/test_elastic_ep.py`
17. `tests/entrypoints/openai/test_realtime_validation.py`
18. `tests/models/multimodal/generation/test_voxtral_realtime.py`
19. `tests/models/multimodal/pooling/test_colmodernvbert.py`
20. `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py`

## 2. 环境记录

- `vllm`：`b31e932`（branch: `v0.17.0`）
- `vllm-ascend`：`e20f0b1a`（branch: `v0.17.0rc1`）
- Python：`3.11.14`
- `torch`：`2.9.0+cpu`
- `torch_npu`：`2.9.0`
- CANN / toolkit：`/usr/local/Ascend/cann-8.5.1`
- 插件启用方式：`VLLM_PLUGINS=ascend`
- 环境脚本：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 模型下载：`HF_ENDPOINT=https://hf-mirror.com`

### 2.1 NPU 选择结论

本轮先用 `npu-smi info` 检查空闲卡，确认 `0/1/2/3/6/7` 无用户进程；但实际运行前还额外用 `torch.npu.mem_get_info()` 验证了可用显存。

关键发现：

- 逻辑空闲不等于 `torch_npu` 视角下“可立即分配内存充足”；
- `ASCEND_RT_VISIBLE_DEVICES=6` 时，`torch.npu.mem_get_info()` 仅看到约 `2.86 GiB` 可用，因此会把很多 `gpu_memory_utilization>=0.8` 的服务类用例直接挡在启动阶段；
- `ASCEND_RT_VISIBLE_DEVICES=0` 时，可用显存约 `29.15 GiB`，更适合单卡分析。

因此：

- 单卡动态分析优先使用 `0` 号卡；
- 多卡类用例需要选取 `0/1` 或 `0/1/2/3` 这类组合，但本轮多数多卡大模型用例在网络/模型资产/规模前置条件处就已能定位真实根因，无需强行跑满整套生成流程。

### 2.2 已消除的环境噪声

在正式归因前，先排除了一个典型伪根因：

- 若把 `VLLM_PLUGINS` 错设为模块名 `vllm_ascend`，插件会被“发现”但不会被“加载”；
- 此时 `current_platform.device_type` 为空，`EngineArgs.create_engine_config()` 会报 `RuntimeError: Device string must not be empty`；
- 这不是测试真实问题，而是插件名过滤错误；正确值必须是 entrypoint 名 `ascend`。

## 3. 批量分析表

说明：

- `Dynamic`：已在 Ascend 环境中实际执行到首个有效失败或有效通过结论；
- `Static`：由于模型下载/远端资源/超大模型/多卡启动代价过高，改为源码与调用链分析，但根因已追到“真实边界”，不是停留在表面报错。

| # | Test File | Evidence | Root Cause | Category | Should Ascend Pass | CI Verdict | Fixable in vllm-ascend |
|---|---|---|---|---|---|---|---|
| 1 | `tests/config/test_config_generation.py` | Dynamic：修正 `VLLM_PLUGINS=ascend` 后，已进入真实测试逻辑；随后阻塞在 `deepseek-ai/DeepSeek-V2-Lite` 配置拉取/SSL 下载阶段。 | 首个真实 blocker 不是 NPU 设备逻辑，而是远端模型配置下载前置条件；文件本身虽然用 `CUDA_VISIBLE_DEVICES` 命名测试，但在 Ascend 上并未暴露插件代码错误。 | `test precondition` | Yes | manual / nightly | No |
| 2 | `tests/engine/test_arg_utils.py` | Dynamic：前 6 个子用例在 Ascend 下连续通过；未出现平台相关断言失败。后续中断来自外部 `KeyboardInterrupt` 噪声，而非测试断言。 | 该文件主体是参数解析/类型工具单测，没有 Ascend 特定分支；未发现 Ascend blocker。 | `compatible / no blocker` | Yes | presubmit | No |
| 3 | `tests/entrypoints/openai/test_default_mm_loras.py` | Static：模块级 `snapshot_download("microsoft/Phi-4-multimodal-instruct")`；服务启动前必须先拿到多模态 base model 和 `speech-lora` 目录。 | 第一真实阻塞点是模型资产获取；完成资产准备后才会进入多模态 LoRA 路径。该路径在 Ascend 上属于高风险适配边界，但当前首个 blocker 仍是外部资源前置条件。 | `test precondition` | Yes (after assets) | nightly / manual | Not for observed blocker |
| 4 | `tests/entrypoints/openai/test_return_tokens_as_ids.py` | Dynamic：纯单元子用例 `test_responses_api_logprobs_with_return_tokens_as_token_ids` 在 Ascend 下通过；在线子用例在 `card6` 上被 `Free memory on device (2.85/29.49 GiB)` 挡住，换到 `card0` 后又卡在模型/LoRA 资产下载的 `ssl.py`。 | 真实 blocker 是资源选择 + 远端资产下载，不是 `return_tokens_as_token_ids` 语义本身；核心 OpenAI response logprobs 逻辑对 Ascend 无阻塞。 | `environment/resource` + `test precondition` | Yes | unit: presubmit; online: nightly | No |
| 5 | `tests/entrypoints/openai/test_video.py` | Static：依赖 `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` 和 3 个外部视频 URL，还断言精确 token usage。 | 真正的首个边界是“远端模型 + 外部视频拉取 + 视频预处理”；这类用例在 Ascend 上不是先受平台算子限制，而是先受网络与外部资源稳定性限制。 | `test precondition` | Likely yes (after assets) | manual / nightly | No |
| 6 | `tests/lora/test_deepseekv2_tp.py` | Static：`DeepSeek-V2-Lite-Chat` + LoRA + TP2/TP4，命中 Ascend LoRA 适配高风险区；与 `case_lora_wrapper_selection_gap.md` 中的 packed/merged wrapper 选型问题同类。 | 真实根因不应先怀疑 upstream `set_lora()`，而应优先怀疑 Ascend 插件对 packed/merged LoRA wrapper 分流是否完整；这是典型插件适配边界问题。 | `vllm-ascend adaptation gap` | Yes | nightly / manual | Yes |
| 7 | `tests/lora/test_qwen3moe_tp.py` | Static：`Qwen/Qwen3-30B-A3B` + LoRA + TP2/TP4，规模更大，仍落在 Ascend LoRA TP 适配边界。 | 与 DeepSeekV2 TP LoRA 同类：真正的高风险点是 Ascend 对 merged/packed/variable-slice LoRA wrapper 选择是否完整，而不是上游 LoRA API 本身。 | `vllm-ascend adaptation gap` | Yes (resource heavy) | nightly / manual | Yes |
| 8 | `tests/model_executor/test_model_load_with_params.py` | Static：3 个 embedding 模型，断言 encoder/pooler/tokenizer 参数加载；只对 ROCm 做了显式跳过，没有 CUDA 专属实现假设。 | 首个真实 blocker 是模型下载/缓存；测试语义本身是参数加载正确性，不依赖 CUDA-only kernel。 | `test precondition` | Yes | nightly | No |
| 9 | `tests/models/language/generation/test_hybrid.py` | Static：覆盖大量 SSM/Hybrid 模型，还混入 `CudagraphDispatcher`、`FULL_CUDA_GRAPH_MODELS`、大规模模型矩阵。 | 该文件把“通用生成正确性”和“CUDA Graph 语义”混在同一文件里。对 Ascend 来说，基础 eager/普通生成子用例可能成立，但整文件不应按原样要求全部通过，尤其 `full_cuda_graph`/graph-dispatch 语义明显偏 CUDA。 | `upstream test hardcoded CUDA` + `runtime feature gap` | Partial | reject / manual | Partially |
| 10 | `tests/models/multimodal/generation/test_whisper.py` | Static：含一个纯单元 `test_parse_language_detection_output`，其余用例依赖 `openai/whisper-large-v3-turbo`、HF 对比和 TP=2 分布式。文件已显式把 worker 多进程方式切到 `spawn`。 | 就代码结构看，首个 blocker 仍是模型资产与大模型运行成本，不是 Ascend 逻辑错误；文件已经主动规避了 fork 问题，说明作者也在绕开非功能性运行时风险。 | `test precondition` | Yes | nightly | No |
| 11 | `tests/plugins_tests/test_io_processor_plugins.py` | Static：单元 `test_loading_missing_plugin` 很轻；核心在线/离线用例依赖 Prithvi 模型和外部 TIFF URL，并要求输出图像 hash 精确匹配。 | 第一真实边界是模型资产与外部 URL 可达性；不是 Ascend 平台逻辑先出错。 | `test precondition` | Yes (after assets) | manual / nightly | No |
| 12 | `tests/v1/ec_connector/unit/test_ec_example_connector.py` | Dynamic：`23 passed, 1 skipped, 1 failed`；失败点为 `AttributeError: 'ECExampleConnector' object has no attribute 'has_caches'`；另有一个子用例被 `@pytest.mark.skipif(not torch.cuda.is_available())` 跳过。 | 真实根因是测试仍在调用已不存在/已重构的旧接口 `has_caches()`；不是 Ascend NPU 逻辑失败。文件里还混有显式 CUDA-only 子用例。 | `upstream test drift` + `upstream test hardcoded CUDA` | Yes (after test adaptation) | presubmit | No |
| 13 | `tests/v1/streaming_input/test_scheduler_streaming.py` | Dynamic：首个失败为 `AssertionError: Encoder-decoder models are expected...`；追代码可见 `create_scheduler()` 里 `model_config` 是 `MagicMock()`，却没有显式设置 `is_encoder_decoder=False`，在新 Scheduler 逻辑中被当作 truthy。 | 根因不是 Ascend，也不是 scheduler 功能退化，而是测试桩没有随着 upstream `Scheduler` 新增的 `is_encoder_decoder` 语义一起更新。 | `upstream test drift` | Yes (after test adaptation) | presubmit | No |
| 14 | `tests/compile/correctness_e2e/test_sequence_parallel.py` | Static：参数集中含 `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`，并直接做 `current_platform.get_device_capability() < (9, 0)`；而 Ascend `get_device_capability()` 返回 `None`，FP8 语义也不是按 CUDA `sm90` 能力线定义。 | 这是典型“把 CUDA capability / FP8 设备语义硬编码进测试”的情况。即便 tiny-random 非 FP8 子集可能有可迁移空间，整文件原样并不适合直接拿来作为 Ascend 全量验证。 | `upstream test hardcoded CUDA` + `runtime feature gap` | Partial (non-FP8 subset only) | reject / manual | Partially |
| 15 | `tests/compile/test_startup.py` | Dynamic：测试甚至未进入文件主体；在 `tests/conftest.py -> vllm.entrypoints.openai.responses.protocol` 导入链中，Pydantic schema 生成长时间卡住并最终中断。 | 当前环境下的首个真实 blocker 是依赖/导入链兼容性问题，而不是 `test_startup.py` 自身的 Ascend 编译逻辑。这意味着本轮无法把责任归到 Ascend compile backend 本身。 | `dependency / environment compatibility` | Unknown | manual | No |
| 16 | `tests/distributed/test_elastic_ep.py` | Static：4 卡、Ray、DeepSeek-V2-Lite-Chat、GSM8K 256 题、弹性扩缩容接口；`vllm-ascend` 代码树内确实已有 `elastic` / `eplb` / `dynamic_eplb` 实现。 | 这类测试不是“插件完全没实现”的场景；真实首个难点是多卡资源、模型资产、Ray 和长时评测链路。它更像重型系统验证，而不是适合快速 CI 的接口守卫。 | `environment/resource` + `test precondition` | Likely yes | manual / nightly | Not for observed blocker |
| 17 | `tests/entrypoints/openai/test_realtime_validation.py` | Static：WebSocket realtime 音频流式验证，依赖 `mistralai/Voxtral-Mini-4B-Realtime-2602`；包含 warm-up、长超时和多阶段连接检查。 | 首个真实 blocker 仍是模型资产和服务启动成本；文件本身未写死 CUDA-only API，但运行成本和网络前置条件都很重。 | `test precondition` | Yes (after assets) | manual / nightly | No |
| 18 | `tests/models/multimodal/generation/test_voxtral_realtime.py` | Static：直接用 `LLM` / `AsyncLLM` 跑 Voxtral realtime，依赖 Mistral tokenizer/HF 模型；无明显 CUDA-only 分支。 | 第一真实边界是模型和 tokenizer 资产准备；文件本身的核心语义是音频流处理和输出一致性，不是 CUDA-only backend 行为。 | `test precondition` | Yes | nightly / manual | No |
| 19 | `tests/models/multimodal/pooling/test_colmodernvbert.py` | Static：`ModernVBERT/colmodernvbert-merged` pooling/token-embed/MaxSim 分数验证；无 CUDA 专属分支。 | 首个 blocker 是模型下载与多模态 pooling 模型启动成本；语义层面属于上游行为契约测试，理论上适合作为 Ascend 功能回归，但不适合无缓存 presubmit。 | `test precondition` | Yes | nightly | No |
| 20 | `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py` | Static：使用 `ExampleHiddenStatesConnector`、自注册可预测 dummy Llama 和 `TinyLlama` tokenizer；与 `kv_connector` 相关，而 `ascend_fix_to_test_mapping.md` 已证明一批 `kv_connector` 单测曾被 Ascend 平台修复解锁。 | 真实首个 blocker 是模型/tokenizer 资产；从代码语义看，这个测试更接近“上游接口契约 + connector 输出正确性”，不是 CUDA-only 行为。 | `test precondition` | Yes | nightly | No |

## 4. 已动态确认的关键根因

### 4.1 `tests/v1/streaming_input/test_scheduler_streaming.py`

真实失败：

- `AssertionError: Encoder-decoder models are expected to implement the multimodal interface with at most one modality.`

根因：

- 测试里的 `create_scheduler()` 使用 `MagicMock()` 伪造 `model_config`；
- 但没有显式设置 `is_encoder_decoder=False`；
- 新版 `Scheduler` 在初始化时读取 `vllm_config.model_config.is_encoder_decoder`；
- `MagicMock` 在布尔上下文中为真，于是错误进入 encoder-decoder 分支并触发断言。

结论：

- 这是 upstream 测试桩漂移，不是 Ascend 平台故障。

### 4.2 `tests/v1/ec_connector/unit/test_ec_example_connector.py`

真实失败：

- `AttributeError: 'ECExampleConnector' object has no attribute 'has_caches'`

根因：

- 测试仍在调用旧接口 `has_caches()`；
- 当前实现只保留了 `has_cache_item()` 等新路径；
- 因此失败所有权在测试代码与接口演进不同步，而不在 Ascend NPU。

附加发现：

- 文件内还有一个子用例被 `@pytest.mark.skipif(not torch.cuda.is_available())` 跳过，属于显式 CUDA-only 测试。

### 4.3 `tests/entrypoints/openai/test_return_tokens_as_ids.py`

已确认：

- 纯单元子用例 `test_responses_api_logprobs_with_return_tokens_as_token_ids` 在 Ascend 下通过。

在线子用例的真实 blocker：

1. 在 `ASCEND_RT_VISIBLE_DEVICES=6` 上，`torch.npu.mem_get_info()` 只看到约 `2.85 GiB` 可用内存，服务启动被 Ascend worker 的 free-memory 检查拒绝；
2. 换到 `ASCEND_RT_VISIBLE_DEVICES=0` 后，启动前移到远端模型/LoRA 资产下载阶段并卡在 `ssl.py`。

结论：

- 当前观察到的 blocker 是资源和网络前置条件，不是该特性与 Ascend 不兼容。

### 4.4 `tests/compile/correctness_e2e/test_sequence_parallel.py`

虽然本轮没有把整套多卡编译链跑穿，但源码已经给出真实根因边界：

- 文件把 FP8 设备能力绑定到 CUDA `sm90` 语义；
- 并直接拿 `current_platform.get_device_capability()` 与 `(9, 0)` 做比较；
- Ascend 平台并不提供 CUDA capability tuple，因此该判断本身就是 CUDA-only 假设；
- 同时 FP8 支持也不是 Ascend 当前应承诺与 CUDA `sm90` 完全同构。

结论：

- 该文件不能整体原样要求 Ascend 通过；最多只能裁切非 FP8、非 CUDA capability 假设的子集。

## 5. 推荐的 CI 分层

### 5.1 推荐加入 presubmit 的高信号子集

- `tests/engine/test_arg_utils.py`
- 适配后的 `tests/v1/streaming_input/test_scheduler_streaming.py`
- 适配后的 `tests/v1/ec_connector/unit/test_ec_example_connector.py`
- `tests/entrypoints/openai/test_return_tokens_as_ids.py::test_responses_api_logprobs_with_return_tokens_as_token_ids`

### 5.2 推荐放入 nightly 的用例

- `tests/model_executor/test_model_load_with_params.py`
- `tests/models/multimodal/generation/test_whisper.py`
- `tests/models/multimodal/pooling/test_colmodernvbert.py`
- `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py`
- `tests/entrypoints/openai/test_return_tokens_as_ids.py`（在线服务子集）

### 5.3 建议保留为 manual / reject 的用例

- `tests/entrypoints/openai/test_video.py`
- `tests/models/language/generation/test_hybrid.py`
- `tests/compile/correctness_e2e/test_sequence_parallel.py`
- `tests/distributed/test_elastic_ep.py`
- `tests/entrypoints/openai/test_default_mm_loras.py`
- `tests/entrypoints/openai/test_realtime_validation.py`
- `tests/models/multimodal/generation/test_voxtral_realtime.py`
- `tests/lora/test_deepseekv2_tp.py`
- `tests/lora/test_qwen3moe_tp.py`

## 6. 最关键的行动项

1. **测试适配优先**：先修复 `test_scheduler_streaming.py` 与 `test_ec_example_connector.py` 这两个明确的 upstream 测试漂移问题；
2. **统一资源筛卡策略**：后续所有单卡 Ascend 动态分析都应先用 `torch.npu.mem_get_info()` 做二次筛卡，而不是只看 `npu-smi` 的“无进程”；
3. **LoRA TP 路径重点关注插件适配**：`test_deepseekv2_tp.py` 与 `test_qwen3moe_tp.py` 优先沿 `case_lora_wrapper_selection_gap.md` 的 wrapper 分流逻辑排查；
4. **不要把 CUDA-only 编译/FP8 语义直接平移到 Ascend**：`test_sequence_parallel.py` 必须裁切，不能整文件生搬到 Ascend CI；
5. **重型多模态 / realtime / elastic 用例先做资产缓存和 nightly 化**：否则分析与 CI 都会被网络和模型下载噪声淹没。
