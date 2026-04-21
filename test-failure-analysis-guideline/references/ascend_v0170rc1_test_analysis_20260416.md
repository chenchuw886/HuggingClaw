# vLLM v0.17.0rc1 上游用例在 Ascend NPU 上的批量分析

更新时间：2026-04-21

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
| 1 | `tests/config/test_config_generation.py` | Dynamic：补跑后已越过模型配置下载，真实失败变为 `AssertionError: Configs with normal CUDA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES="" should be equivalent`；差异集中在 `ascend_compilation_config.enable_npugraph_ex`。 | 根因不是网络，而是 Ascend 配置生成对“空字符串 `CUDA_VISIBLE_DEVICES`”路径与“未设置”路径给出了不同 `additional_config`，导致该 upstream 等价性断言失效。 | `vllm-ascend config divergence` | Yes (after config alignment) | nightly / targeted | Yes |
| 2 | `tests/engine/test_arg_utils.py` | Dynamic：前 6 个子用例在 Ascend 下连续通过；未出现平台相关断言失败。后续中断来自外部 `KeyboardInterrupt` 噪声，而非测试断言。 | 该文件主体是参数解析/类型工具单测，没有 Ascend 特定分支；未发现 Ascend blocker。 | `compatible / no blocker` | Yes | presubmit | No |
| 3 | `tests/entrypoints/openai/test_default_mm_loras.py` | Dynamic：补跑后服务已进入 engine-core 初始化，首个有效失败为 LoRA 路径 OOM：`selected_loras = lora_b_weights[lora_indices_tensor].to(...)` 触发 `RuntimeError: NPU out of memory. Tried to allocate 20.00 GiB`。 | 根因已不是资产下载，而是 Phi-4 多模态 LoRA 启动/profile run 在当前 Ascend 显存预算下发生真实 NPU OOM；失败点位于 LoRA 扩展/索引张量搬运路径。 | `runtime memory pressure` + `vllm-ascend adaptation risk` | Maybe (after memory/shape tuning) | manual / nightly | Partially |
| 4 | `tests/entrypoints/openai/test_return_tokens_as_ids.py` | Dynamic：纯单元子用例 `test_responses_api_logprobs_with_return_tokens_as_token_ids` 在 Ascend 下通过；在线 completion 子用例补跑后不再停在下载阶段，而是服务端 500，内核首错为 `RuntimeError: the first dimension of x, y, indices should be same`。 | 根因不是网络，而是在线推理路径中 prompt logprobs / LoRA logits 相关张量维度不一致，导致 Ascend 执行期崩溃并向 OpenAI server 冒泡为 `EngineDeadError`。 | `vllm-ascend runtime bug` | Unit yes; online no | unit: presubmit; online: manual / nightly | Yes |
| 5 | `tests/entrypoints/openai/test_video.py` | Dynamic：在独占空闲 `1` 号卡上再次单独重跑后，仍未首先暴露外部视频下载问题；当前可见首个失败是本地 server 对 `127.0.0.1:<port>/health` 的请求直接 `Connection refused`，随后 `_wait_for_server` 被中断。 | 现阶段真实边界已稳定前移到“本地 OpenAI server 根本未成功 bind/ready”，而不是单纯网络资源；但该日志仍未带出子进程 stderr/traceback，因此深一层启动异常尚未被当前工件捕获。 | `runtime startup failure (child traceback missing)` | Unknown | manual | Unknown |
| 6 | `tests/lora/test_deepseekv2_tp.py` | Static：`DeepSeek-V2-Lite-Chat` + LoRA + TP2/TP4，命中 Ascend LoRA 适配高风险区；与 `case_lora_wrapper_selection_gap.md` 中的 packed/merged wrapper 选型问题同类。 | 真实根因不应先怀疑 upstream `set_lora()`，而应优先怀疑 Ascend 插件对 packed/merged LoRA wrapper 分流是否完整；这是典型插件适配边界问题。 | `vllm-ascend adaptation gap` | Yes | nightly / manual | Yes |
| 7 | `tests/lora/test_qwen3moe_tp.py` | Static：`Qwen/Qwen3-30B-A3B` + LoRA + TP2/TP4，规模更大，仍落在 Ascend LoRA TP 适配边界。 | 与 DeepSeekV2 TP LoRA 同类：真正的高风险点是 Ascend 对 merged/packed/variable-slice LoRA wrapper 选择是否完整，而不是上游 LoRA API 本身。 | `vllm-ascend adaptation gap` | Yes (resource heavy) | nightly / manual | Yes |
| 8 | `tests/model_executor/test_model_load_with_params.py` | Dynamic：补跑后已进入真实模型加载/预热流程；日志中先出现 `Cannot run aclop operators during NPU graph capture ... Fill ...`，随后框架打印 `Falling back to eager warmup after NPU graph capture failure.`，最终整条 pytest 被外层 `timeout` 截断。 | 目前可见的最深有效失败点不是下载，而是 Ascend 图捕获预热路径对 `Fill` 等 ACL op 不兼容；最终用例是否还能继续失败被外层超时遮蔽，但已足以说明该文件的 blocker 前移到 NPU graph warmup。 | `vllm-ascend compile/warmup fragility` + `timeout masking` | Unknown | nightly / manual | Yes |
| 9 | `tests/models/language/generation/test_hybrid.py` | Static：覆盖大量 SSM/Hybrid 模型，还混入 `CudagraphDispatcher`、`FULL_CUDA_GRAPH_MODELS`、大规模模型矩阵。 | 该文件把“通用生成正确性”和“CUDA Graph 语义”混在同一文件里。对 Ascend 来说，基础 eager/普通生成子用例可能成立，但整文件不应按原样要求全部通过，尤其 `full_cuda_graph`/graph-dispatch 语义明显偏 CUDA。 | `upstream test hardcoded CUDA` + `runtime feature gap` | Partial | reject / manual | Partially |
| 10 | `tests/models/multimodal/generation/test_whisper.py` | Dynamic：补跑代表性单卡子集后，失败不再停在模型下载，而是在 `VllmConfig` 校验期触发 `Assertion failed, When enabling VLLM_COMPILE aclgraph, please make sure compilation_config.mode == CompilationMode.VLLM_COMPILE and compilation_config.cudagraph_mode == CUDAGraphMode.VLLM_COMPILE`。 | 根因是 Ascend 侧启用了 ACL graph / compile 相关开关，但组装出的 `compilation_config` 与 `VLLM_COMPILE` 约束不一致，属于配置注入/编译模式联动错误。 | `vllm-ascend config bug` | No (current env) | nightly / manual | Yes |
| 11 | `tests/plugins_tests/test_io_processor_plugins.py` | Dynamic：补跑后在线/离线 Prithvi 子用例都在模型架构检查阶段失败，错误为 `ModuleNotFoundError: No module named 'terratorch'`，并伴随 `Model architectures ['Terratorch'] failed to be inspected`。 | 根因不是网络或 TIFF URL，而是测试环境缺少 Prithvi/Terratorch 所需的可选依赖，导致模型注册/检查在启动前失败。 | `dependency / environment compatibility` | Yes (after deps) | manual / nightly | No |
| 12 | `tests/v1/ec_connector/unit/test_ec_example_connector.py` | Dynamic：`23 passed, 1 skipped, 1 failed`；失败点为 `AttributeError: 'ECExampleConnector' object has no attribute 'has_caches'`；另有一个子用例被 `@pytest.mark.skipif(not torch.cuda.is_available())` 跳过。 | 真实根因是测试仍在调用已不存在/已重构的旧接口 `has_caches()`；不是 Ascend NPU 逻辑失败。文件里还混有显式 CUDA-only 子用例。 | `upstream test drift` + `upstream test hardcoded CUDA` | Yes (after test adaptation) | presubmit | No |
| 13 | `tests/v1/streaming_input/test_scheduler_streaming.py` | Dynamic：首个失败为 `AssertionError: Encoder-decoder models are expected...`；追代码可见 `create_scheduler()` 里 `model_config` 是 `MagicMock()`，却没有显式设置 `is_encoder_decoder=False`，在新 Scheduler 逻辑中被当作 truthy。 | 根因不是 Ascend，也不是 scheduler 功能退化，而是测试桩没有随着 upstream `Scheduler` 新增的 `is_encoder_decoder` 语义一起更新。 | `upstream test drift` | Yes (after test adaptation) | presubmit | No |
| 14 | `tests/compile/correctness_e2e/test_sequence_parallel.py` | Static：参数集中含 `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`，并直接做 `current_platform.get_device_capability() < (9, 0)`；而 Ascend `get_device_capability()` 返回 `None`，FP8 语义也不是按 CUDA `sm90` 能力线定义。 | 这是典型“把 CUDA capability / FP8 设备语义硬编码进测试”的情况。即便 tiny-random 非 FP8 子集可能有可迁移空间，整文件原样并不适合直接拿来作为 Ascend 全量验证。 | `upstream test hardcoded CUDA` + `runtime feature gap` | Partial (non-FP8 subset only) | reject / manual | Partially |
| 15 | `tests/compile/test_startup.py` | Dynamic：测试甚至未进入文件主体；在 `tests/conftest.py -> vllm.entrypoints.openai.responses.protocol` 导入链中，Pydantic schema 生成长时间卡住并最终中断。 | 当前环境下的首个真实 blocker 是依赖/导入链兼容性问题，而不是 `test_startup.py` 自身的 Ascend 编译逻辑。这意味着本轮无法把责任归到 Ascend compile backend 本身。 | `dependency / environment compatibility` | Unknown | manual | No |
| 16 | `tests/distributed/test_elastic_ep.py` | Static：4 卡、Ray、DeepSeek-V2-Lite-Chat、GSM8K 256 题、弹性扩缩容接口；`vllm-ascend` 代码树内确实已有 `elastic` / `eplb` / `dynamic_eplb` 实现。 | 这类测试不是“插件完全没实现”的场景；真实首个难点是多卡资源、模型资产、Ray 和长时评测链路。它更像重型系统验证，而不是适合快速 CI 的接口守卫。 | `environment/resource` + `test precondition` | Likely yes | manual / nightly | Not for observed blocker |
| 17 | `tests/entrypoints/openai/test_realtime_validation.py` | Dynamic：在独占空闲 `2` 号卡上重跑后，已越过此前的低显存门槛并进入真实测试执行；当前首错变为 teardown 路径 `cleanup_dist_env_and_memory` 调用 `torch._C._host_emptyCache()`，触发 `AttributeError`。 | 根因不再是显存不足，而是清理阶段直接调用当前 torch 构建中不存在的 `_host_emptyCache`；属于运行时清理兼容性问题。 | `dependency / runtime compatibility` | Yes (after cleanup fix) | manual / nightly | No |
| 18 | `tests/models/multimodal/generation/test_voxtral_realtime.py` | Dynamic：在独占空闲 `3` 号卡上重跑后，已越过此前的显存门槛并完成 engine 启动前半段；真实首错前移到模型构建阶段的 `NotImplementedError`：`AscendAttentionBackend` 尚不支持该 whisper block-pooling attention backend。 | 根因不再是显存不足，而是 Voxtral realtime 依赖的 whisper/attention backend 在 Ascend 上尚未实现；外层 `Engine core initialization failed` 只是包装层。 | `vllm-ascend feature gap` | No (current backend) | manual / nightly | Yes |
| 19 | `tests/models/multimodal/pooling/test_colmodernvbert.py` | Dynamic：补跑后 4 个子用例均在 `ModelConfig` 创建期失败，关键报错为 `The checkpoint you are trying to load has model type 'modernvbert' but Transformers does not recognize this architecture.`；当前环境 `transformers==4.57.6`。 | 根因已不是网络，而是测试环境中的 `transformers` 版本/注册表无法识别 `modernvbert`，与 vLLM 文档已声明支持的 `ColModernVBERT` 资产形成依赖错位。 | `dependency / environment compatibility` | Yes (after dependency alignment) | nightly | No |
| 20 | `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py` | Dynamic：补跑后 engine 已启动，但测试在 connector / IPC 交互阶段长时间无结果，最终在 `zmq/backend/cython/_zmq.py:179` 触发 `KeyboardInterrupt`。 | 当前最深有效失败点是 hidden-state connector 集成链路挂起，而不是下载或 CUDA-only 限制；日志尚未给出更早异常，因此暂归为测试/运行时集成挂起。 | `runtime/test integration hang` | Unknown | nightly / manual | Unknown |

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
2. 换到 `ASCEND_RT_VISIBLE_DEVICES=0` 并解决镜像下载后，服务端进一步前移到真实运行期错误：`RuntimeError: the first dimension of x, y, indices should be same`；
3. 该异常随后被封装成 `EngineDeadError` / OpenAI `500 InternalServerError` 暴露给测试。

结论：

- 该文件不能再简单归因为资源或网络；在线子集已经暴露出真实 Ascend 运行期维度错误，而纯单元子集仍是可通过的。

### 4.4 `tests/config/test_config_generation.py`

补跑后已越过模型配置下载，真实失败为：

- `AssertionError: Configs with normal CUDA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES="" should be equivalent`

根因：

- 该测试用同一组 `EngineArgs` 分别在“正常 `CUDA_VISIBLE_DEVICES`”与“空字符串 `CUDA_VISIBLE_DEVICES`”下生成配置；
- Ascend 路径下两次生成出的 `additional_config.ascend_compilation_config.enable_npugraph_ex` 不一致；
- 因此失败所有权落在 Ascend 配置生成逻辑，而不是网络、模型下载或通用设备选择逻辑。

### 4.5 `tests/entrypoints/openai/test_default_mm_loras.py`

补跑后首个有效失败已前移到 engine core：

- `RuntimeError: NPU out of memory. Tried to allocate 20.00 GiB`

关键栈信息：

- 失败点出现在 LoRA 权重扩展路径 `selected_loras = lora_b_weights[lora_indices_tensor].to(...)`；
- 外层的 `Server exited unexpectedly` 只是包装层，真正根因是 Phi-4 多模态 LoRA profile/warmup 阶段触发大块显存申请失败。

### 4.6 `tests/entrypoints/openai/test_video.py`

在独占空闲 `1` 号卡上再次单独重跑后已确认：

- 日志中最早的具体失败不再是下载或外部视频 URL；
- fixture 在轮询本地 `127.0.0.1:<port>/health` 时直接收到 `Connection refused`；
- 随后 `_wait_for_server` 路径被 `KeyboardInterrupt` 打断。

结论：

- 该文件当前首个稳定可见 blocker 是“测试 server 根本没有成功监听端口”；
- 但现有日志没有带出被拉起子进程的 stderr/traceback，因此还不能把责任继续下钻到更具体的 engine 内部异常。

### 4.7 `tests/model_executor/test_model_load_with_params.py`

补跑后已确认：

- 日志先出现 `Cannot run aclop operators during NPU graph capture ... Fill ...`；
- 随后框架打印 `Falling back to eager warmup after NPU graph capture failure.`；
- 但整个 pytest 进程在更晚阶段被外层 `timeout` 截断，因此最终失败点仍有一层超时遮蔽。

结论：

- 该文件不能再归因为“模型未下载”；
- 当前最深有效根因是 Ascend 图捕获预热链对 ACL op 的兼容性脆弱，后续还需要更长超时窗口或更小测试切片来继续下钻。

### 4.8 `tests/models/multimodal/generation/test_whisper.py`

补跑代表性单卡子集后，真实失败为：

- `Assertion failed, When enabling VLLM_COMPILE aclgraph, please make sure compilation_config.mode == CompilationMode.VLLM_COMPILE and compilation_config.cudagraph_mode == CUDAGraphMode.VLLM_COMPILE`

根因：

- Ascend 侧 ACL graph / compile 相关配置被启用；
- 但最终进入 `VllmConfig` 校验的 `compilation_config` 组合不自洽；
- 因此失败属于 Ascend 配置注入问题，而不是 Whisper 模型资产或 fork/spawn 运行方式本身。

### 4.9 `tests/plugins_tests/test_io_processor_plugins.py`

补跑后在线/离线 Prithvi 子用例共同暴露出更深 blocker：

- `ModuleNotFoundError: No module named 'terratorch'`

结论：

- 该文件首个失败点是 Terratorch 可选依赖缺失；
- 因此它不是“外部 TIFF URL 不稳定”的网络类问题，而是测试环境缺少对应模型插件依赖。

### 4.10 `tests/models/multimodal/pooling/test_colmodernvbert.py`

补跑后 4 个子用例一致失败：

- `The checkpoint you are trying to load has model type 'modernvbert' but Transformers does not recognize this architecture.`

补充环境信息：

- 当前环境 `transformers==4.57.6`；
- vLLM 代码树中已包含 `colmodernvbert` 相关配置与模型注册，因此当前更像运行环境依赖版本/注册表与测试资产定义之间的不匹配。

### 4.11 `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py`

补跑后可见：

- engine 已成功启动；
- 但测试随后长时间停在 connector / IPC 交互阶段；
- 最终在 `zmq/backend/cython/_zmq.py:179` 被 `KeyboardInterrupt` 打断。

结论：

- 该文件当前最深有效失败点是集成链路挂起；
- 已不能再归因于模型/tokenizer 下载前置条件。

### 4.12 `tests/entrypoints/openai/test_realtime_validation.py`

在独占空闲 `2` 号卡上单独重跑后已确认：

- 测试能够越过此前的 free-memory 守卫并进入真实执行；
- 当前首个有效异常出现在 teardown 清理阶段；
- `cleanup_dist_env_and_memory` 调用 `torch._C._host_emptyCache()`，但当前 torch 构建未提供该符号，触发 `AttributeError`。

结论：

- 该文件当前首个真实 blocker 已不是网络，也不是显存不足；
- 真实问题前移到分布式/显存清理兼容性路径。

### 4.13 `tests/models/multimodal/generation/test_voxtral_realtime.py`

在独占空闲 `3` 号卡上单独重跑后可见：

- `Transformers does not recognize this architecture` 只是前置警告，vLLM 随后完成了 `Resolved architecture: VoxtralRealtimeGeneration`；
- 测试已越过此前的 free-memory 校验；
- 真正首错前移到模型构建阶段：`NotImplementedError: <class 'vllm_ascend.attention.attention_v1.AscendAttentionBackend'> is not yet supported.`；
- 外层随后报出 `RuntimeError: Engine core initialization failed`。

结论：

- 该文件也不能再归因为网络或显存问题；
- 当前最深有效根因是 Ascend attention backend 功能缺口。

### 4.14 `tests/compile/correctness_e2e/test_sequence_parallel.py`

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

- `tests/config/test_config_generation.py`（修复 Ascend 配置分歧后）
- `tests/model_executor/test_model_load_with_params.py`
- `tests/models/multimodal/generation/test_whisper.py`
- `tests/models/multimodal/pooling/test_colmodernvbert.py`（补齐依赖版本后）
- `tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py`

### 5.3 建议保留为 manual / reject 的用例

- `tests/entrypoints/openai/test_video.py`
- `tests/models/language/generation/test_hybrid.py`
- `tests/compile/correctness_e2e/test_sequence_parallel.py`
- `tests/distributed/test_elastic_ep.py`
- `tests/entrypoints/openai/test_default_mm_loras.py`
- `tests/entrypoints/openai/test_return_tokens_as_ids.py`（在线服务子集）
- `tests/entrypoints/openai/test_realtime_validation.py`
- `tests/models/multimodal/generation/test_voxtral_realtime.py`
- `tests/lora/test_deepseekv2_tp.py`
- `tests/lora/test_qwen3moe_tp.py`

## 6. 最关键的行动项

1. **测试适配优先**：先修复 `test_scheduler_streaming.py` 与 `test_ec_example_connector.py` 这两个明确的 upstream 测试漂移问题；
2. **统一资源筛卡策略**：后续所有单卡 Ascend 动态分析都应先用 `torch.npu.mem_get_info()` 做二次筛卡，而不是只看 `npu-smi` 的“无进程”；
3. **优先修复 Ascend 配置分歧**：`test_config_generation.py` 与 `test_whisper.py` 已表明 `ascend_compilation_config` / `VLLM_COMPILE` 相关配置联动存在真实问题；
4. **LoRA 与在线服务路径继续重点关注插件适配**：`test_default_mm_loras.py` 和在线 `test_return_tokens_as_ids.py` 都已暴露到真实运行期问题；`test_deepseekv2_tp.py`、`test_qwen3moe_tp.py` 仍应沿 `case_lora_wrapper_selection_gap.md` 的 wrapper 分流逻辑排查；
5. **不要把 CUDA-only 编译/FP8 语义直接平移到 Ascend**：`test_sequence_parallel.py` 必须裁切，不能整文件生搬到 Ascend CI；
6. **补齐模型可选依赖与版本矩阵**：`test_io_processor_plugins.py` 需要 `terratorch`，`test_colmodernvbert.py` 需要与 `modernvbert` 对齐的 `transformers`/注册表版本；
7. **重型 multimodal / realtime / elastic 用例继续 nightly 化**：它们现在虽然不全是“网络问题”，但仍带有 server startup、IPC 挂起、长时资源占用等系统级噪声。
