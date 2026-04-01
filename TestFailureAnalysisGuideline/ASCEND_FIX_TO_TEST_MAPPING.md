# `vllm-ascend` 具体改动点 -> 用例恢复映射表

更新时间：2026-04-01

## 1. 目的

本文件只回答一个问题：

> 哪一个具体代码改动点，对应让哪些用例跑通，或者至少越过了之前的首个阻塞点。

为避免“相关但未证实”的映射混入，本文把结论分成两类：

1. **已明确跑通映射**：改动后，相关用例/参数化子例已实际补跑通过；
2. **已明确推进但未全绿**：改动后，旧首错消失，但文件级别/全量用例还存在下游阻塞。

本轮动态补跑统一要求：**单用例、单空闲 NPU、独立进程**。

## 1.1 本文“修复意义”分析绑定的代码提交

下文关于“修复意义”的深入分析，默认绑定到这次实际代码提交：

- `vllm-ascend` commit: `d49937c9451cb898e804a4bad0e54a2ecc34531e`

之所以显式写出 commit，是因为“哪条用例恢复映射成立”和“这条修复在架构上的真实价值是什么”都必须回到实际 diff 分析，而不能只根据补跑现象倒推。

---

## 2. 已明确跑通映射

| 改动文件 | 具体代码点 | 改动内容 | 改后跑通的用例 |
|---|---|---|---|
| `vllm_ascend/platform.py` | `NPUPlatform.check_and_update_config()` | 当 `device_config.device_type != "npu"` 时直接早返回；`model_config is None` 时跳过 Ascend 专属配置更新，避免 CPU/config-only 场景被平台插件污染。 | `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py` (`1 passed`)、`test_error_propagation.py` (`2 passed`)、`test_invalid_blocks_correctness.py` (`3 passed`)、`test_kv_load_failure_recovery.py` (`11 passed`)、`test_decode_bench_connector.py` (`8 passed`)、`test_offloading_connector.py` (`10 passed`)、`test_remote_decode_lifecycle.py` (`4 passed`)、`test_remote_prefill_lifecycle.py` (`6 passed`) |
| `vllm_ascend/ascend_config.py` | `AscendConfig.__init__()` 中 `model_config` 相关分支 | 对 `model_config.is_deepseek_mla`、`hf_text_config`、`enable_kv_nz` 等访问增加显式保护，避免纯配置场景崩溃。 | `tests/v1/kv_connector/unit/test_config.py` (`6 passed`) |
| `vllm_ascend/platform.py` | `_get_npu_smi_hbm_capacity_mb()` + `get_device_total_memory()` | 补齐 `NPUPlatform.get_device_total_memory()`，并优先走 `npu-smi` 获取总显存，避免在 fork 前提前初始化 `torch.npu`。 | `tests/v1/engine/test_engine_args.py` (`2 passed, 1 skipped`) |
| `vllm_ascend/platform.py` | `_ensure_ascend_worker_multiproc_method()` + `pre_register_and_update()` / `check_and_update_config()` | 当用户未显式设置 `VLLM_WORKER_MULTIPROC_METHOD` 时，Ascend 默认改用 `spawn`，但仍保留用户手动指定的 `fork/spawn`。该改动用于规避 `torch-npu` / ACL runtime 的 fork-unsafe 行为。 | `tests/lora/test_qwenvl.py`：先前稳定出现的 `Invalid thread pool!` 首错已消失；我们已在“单用例、单空闲 NPU、独立进程”条件下独立补跑通过 `test_qwen2vl_lora`、`test_qwen2vl_lora_beam_search`，且用户随后按同一口径手动复跑整文件确认 `6 passed`。 |
| `vllm_ascend/lora/utils.py` | `Ascend*WithLoRA.can_replace_layer()` + `AscendMergedColumnParallelLinearVariableSliceWithLoRA` | 恢复与 upstream 对齐的 LoRA wrapper 分流规则：不再把 `AscendMergedColumnParallelLinear` 粗暴折叠为单一 wrapper，而是按 `packed_modules_list` 长度与 `output_sizes` 判定单 packed / 2-slice / variable-slice 语义。 | 已由用户在真实环境验证 `tests/lora/test_add_lora.py` 修复通过；同时它也是 `tests/lora/test_qwenvl.py` 能稳定收敛的重要前提之一。 |
| `vllm_ascend/lora/punica_npu.py` | `_get_token_lora_indices()`、`_shrink_decode()`、`_expand_decode()`、`_expand_slice_decode()` | decode 路径不再直接使用完整 `token_lora_indices`，而是按当前 `x.size(0)` 收窄，修复多模态 LoRA 场景下 `x/y/indices` 第一维不一致。 | `tests/lora/test_qwenvl.py`：旧的 LoRA decode 维度错配首错已被消除；结合上面的默认 `spawn` 修复后，用户已按“单用例、单空闲 NPU、独立进程”口径手动复跑整文件确认 `6 passed`。 |
| `vllm_ascend/worker/model_runner_v1.py` | `_torch_cuda_wrapper()` | 异常清理后改为 `raise` 原始异常，而不是统一包装成 `RuntimeError("NPUModelRunner init failed ...")`，从而保留 upstream 断言所需的异常类型。 | `tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT-pooling]`、`[...ENTRYPOINT-spec_dec]`、`[...FQCN-pooling]`、`[...FQCN-spec_dec]`、`[...CLASS-pooling]`、`[...CLASS-spec_dec]` 已在空闲 NPU 0-5 上逐例独立补跑通过。 |

---

## 2.1 各修复点的“修复意义”深入说明

这一节不重复“哪个测试通过了”，而是回答另一个更重要的问题：

> 这些修复对 `vllm-ascend` 作为平台插件本身，分别消除了什么结构性风险，恢复了什么语义边界，又降低了哪一类长期维护成本。

### 2.1.1 `platform.py::check_and_update_config()` 的 CPU / no-model 早返回

这条修复的核心价值，不是“让几个测试安静下来”，而是**重新划定了 Ascend 平台插件的生效边界**。

从实际代码看，`NPUPlatform.check_and_update_config()` 在进入 Ascend 专属逻辑前，新增了两层入口过滤：

- `device_config.device_type != "npu"` 时直接返回；
- `model_config is None` 时直接返回。

这两层过滤非常关键，因为它们把插件生效范围从“任何拿到 `VllmConfig` 的场景”收窄为“确实在 NPU 上、且已经有模型语义的场景”。这解决的不是单点异常，而是一个更深层的问题：**Ascend 插件此前把自己从平台适配层，扩张成了会修改 upstream 通用配置路径的全局副作用层。**

其长期价值主要有三点：

1. 恢复插件边界  
   平台插件应该只在本平台语义成立时介入；CPU 路径和 config-only 路径本来应保持 upstream 原义。早返回后，Ascend 不再“越界解释”这些配置对象。
2. 降低与 upstream 演进的耦合  
   upstream 0.17.0rc1 中大量测试与工具路径合法使用 `VllmConfig()` 或 `model_config=None`。如果 Ascend 继续在这些路径上注入 NPU 专属修正，那么每次 upstream 新增一个纯配置测试，都可能被 Ascend 误伤。早返回实质上是在切断这种脆弱耦合。
3. 把“插件装上即改变默认行为”的风险降到最低  
   这条修复让 `pip install vllm-ascend` 的副作用更可控。插件存在本身，不再等价于“所有 upstream 配置对象都可能被 NPU 逻辑改写”。

因此，这条修复的真正意义是：**它把 `vllm-ascend` 从一个会污染非 Ascend 路径的 intrusive plugin，收敛回一个只在本设备语义成立时才生效的平台插件。**

### 2.1.2 `ascend_config.py` 中 `model_config` 显式保护

这条修复的价值，在于**把隐式前提改成显式契约**。

提交前，`AscendConfig.__init__()` 内部多处直接访问：

- `model_config.is_deepseek_mla`
- `model_config.hf_text_config`
- `enable_kv_nz` 对应的模型能力分支

这些访问默认假设“只要进入 AscendConfig，`model_config` 一定有效”。但这个假设在 upstream 里并不成立，因为 `VllmConfig` 合法支持 `model_config=None`。此次修复不是简单补几个 `if None`，而是把代码真实依赖的语义前提写清楚了：

- 对可选路径，只有在 `model_config` 存在时才访问模型语义；
- 对 `enable_kv_nz` 这种强依赖模型形态的功能，明确抛出“需要合法 model_config”的错误。

这有两个深层价值：

1. 从“偶然不崩”升级到“失败语义可解释”  
   以前的问题是，config-only 场景可能在 unrelated 的属性访问点炸掉；现在变成了“该功能本来就要求模型语义”的显式契约失败。前者是插件泄漏实现细节，后者是接口条件清晰。
2. 避免 Ascend 功能把自身前置条件偷偷外溢到整个配置系统  
   `enable_kv_nz`、PD TP ratio、DeepSeek MLA 等都是很强的模型形态假设。修复后，这些假设被限制在真正依赖它们的功能域内，而不是隐式污染所有 `AscendConfig` 初始化路径。

对插件维护来说，这意味着：**未来继续扩展 Ascend 专属配置时，代码会更自然地分成“可在无模型场景安全存在的配置”和“必须绑定模型语义的配置”两类，维护成本和回归风险都显著下降。**

### 2.1.3 `platform.py::get_device_total_memory()` 与 `npu-smi` 优先路径

这条修复的深层意义，不只是“补了一个 API”，而是**把设备总显存查询从 runtime-sensitive 的 Python 设备初始化路径中解耦出来**。

提交后的实现先尝试：

- `npu-smi info -t memory -i <id> -c 0`

失败时才回退到：

- `torch.npu.get_device_properties(device_id)`

这背后的价值在 Ascend 上尤其大，因为 `torch.npu` 的很多查询不是纯读操作，而是可能隐式触发 runtime 初始化。对于会经历 `fork`/`spawn`、或在 worker 建立前需要先做资源判断的路径，这种“读取信息即带副作用”的行为非常危险。

因此这条修复的真正收益是：

1. 降低启动阶段的 runtime 提前初始化风险  
   把“查询总显存”尽量降级为一次外部工具读取，避免在不该初始化 ACL / torch-npu 的时机碰设备 runtime。
2. 让资源探测更接近平台能力查询，而不是执行态副作用  
   `get_device_total_memory()` 从“顺手摸 runtime 对象”变成“平台层的设备属性查询”，抽象层次更正确。
3. 为后续多进程、资源调度、预检查逻辑提供更稳定基石  
   只要某条路径需要在 worker 真正拉起前估算容量，这种低副作用探测就比直接问 `torch.npu` 稳定得多。

换句话说，这条修复提升的不是单个接口，而是**Ascend 平台层“先探测、后初始化”的工程纪律**。

### 2.1.4 默认将 `VLLM_WORKER_MULTIPROC_METHOD` 设为 `spawn`

这条修复的关键价值，是**把一个依赖使用者经验的环境约束，提升为平台默认策略**。

提交前，Ascend 用户如果没有主动设置 `VLLM_WORKER_MULTIPROC_METHOD`，整个系统会沿用 upstream 或环境默认多进程方式；而对 `torch-npu` / ACL runtime 来说，`fork` 不是安全默认值。于是问题表现为：

- 代码本身可能完全正确；
- 失败却发生在子进程初始化、线程池、runtime 句柄继承等“环境层”。

提交后的 `_ensure_ascend_worker_multiproc_method()` 做了两件同时成立的事：

- 当用户未设置时，默认切到 `spawn`；
- 当用户已显式设置时，保留用户选择。

这条设计非常重要，因为它不是简单“强推 spawn”，而是在“平台安全默认值”和“用户控制权”之间取得了正确平衡。

它的长期价值主要体现在：

1. 把非 fork-safe 运行时约束内建到平台层  
   这不是测试 patch，而是把 Ascend runtime 的真实约束编码成默认行为，减少“只有经验丰富的用户才知道该怎么设环境变量”的隐性门槛。
2. 大幅降低偶发性、环境依赖型故障  
   `Invalid thread pool!` 这类问题最难维护，因为它们往往和代码逻辑本身弱相关，却会持续污染回归结果。默认 `spawn` 等于从平台入口处切掉了一整类不稳定性来源。
3. 保持与 upstream 兼容的可操作性  
   没有强行覆盖用户显式指定的 `fork/spawn`，意味着平台给出安全默认，但不剥夺调试、性能实验和特殊部署场景下的主动控制。

所以这条修复的本质是：**把 Ascend 平台“已知不安全的运行方式”从用户自觉规避，升级为插件默认防护。**

### 2.1.5 `lora/utils.py` 的 wrapper 选择修复

这条修复的价值非常高，因为它修复的不是简单 shape 问题，而是**Ascend 插件对 upstream LoRA 类型系统的一处语义塌缩**。

从实际 diff 看，修复前 Ascend 对 `AscendMergedColumnParallelLinear` 的 LoRA wrapper 分流过于粗糙，丢掉了 upstream 中非常关键的判定信息：

- `packed_modules_list == 1`
- `packed_modules_list == 2`
- `packed_modules_list >= 3`
- 某些 variable-slice 场景还要结合 `output_sizes`

修复后，Ascend 恢复了与 upstream 对齐的分流结构：

- 单 packed merged linear 走 `ColumnParallelLinearWithLoRA` 语义；
- 2-slice merged linear 走 `MergedColumnParallelLinearWithLoRA`；
- 3+ slice 或 variable-slice 场景走新的 `AscendMergedColumnParallelLinearVariableSliceWithLoRA`。

它的深层意义在于：

1. 恢复“同一底层层类型，不同 LoRA 语义必须分流”的架构事实  
   upstream 之所以有多个 wrapper，不是实现偏好，而是因为不同 packed 形态对应不同权重组织和 `set_lora()` 语义。Ascend 之前把这些语义压扁了，导致上层看似“类型兼容”，实际运行时结构错位。
2. 防止插件把 upstream 的抽象层次打平  
   这个问题最危险的地方在于：报错通常落在 upstream LoRA 层，看起来像 upstream bug。修复后的价值之一，就是重新确保插件不会悄悄破坏 upstream 已经建立好的类型分发协议。
3. 提升模型覆盖面，不再靠偶然兼容  
   ChatGLM、Qwen2-VL 一类模型里，merged linear 与 LoRA 映射并不是一一等价的。恢复正确分流后，插件对 packed / merged / variable-slice 模型的兼容性是结构性提高，而不是只修某个模型名。

这条修复可以视为：**Ascend LoRA 适配从“按类名粗分类”升级回“按真实 packed 语义分类”。**

### 2.1.6 `ops/linear.py` 中移除对 `output_sizes` 的默认覆写

这处 diff 量不大，但价值并不浅。它的意义在于：**撤销一次会污染类型语义边界的错误补偿。**

提交前，`AscendColumnParallelLinear.__init__()` 在 `output_sizes is None` 时会强行做：

- `output_sizes = [output_size]`

但这并不会真正把该对象变成 upstream 语义上的“merged/packed layer”；反而会让某些后续逻辑误以为它具备 `output_sizes` 相关结构信息。也就是说，这种写法属于“通过伪造成员形态来掩盖上游 wrapper 选型不准确”的方向。

此次删除它的价值有三点：

1. 保住普通 `ColumnParallelLinear` 与 packed / merged linear 的边界  
   普通列并行层不应因为插件方便就被伪装成拥有 packed 元信息的层。
2. 避免错误修复方向固化进基类  
   一旦这种“补成员变量兜底”的思路保留下来，后续更多 wrapper / shape 判定就会建立在被污染的对象语义上，问题会越修越隐蔽。
3. 和上一条 wrapper 分流修复形成闭环  
   `lora/utils.py` 负责把正确 wrapper 选出来；`ops/linear.py` 这处删除则保证底层层对象本身不再提供误导性的假结构。两者一起，才是真正恢复 LoRA 语义边界。

因此，这条改动虽然没有单独宣称修复某个用例，但它在工程上承担的是**“防止错误补丁模型继续扩散”**的作用。

### 2.1.7 `punica_npu.py` 的 decode 索引收窄

这条修复的价值在于：**把 decode 阶段真正参与计算的 token 子集，与 LoRA 索引元数据重新对齐。**

修复前，decode 路径直接把完整 `token_lora_indices` 传给：

- `bgmv_shrink`
- `bgmv_expand`
- `bgmv_expand_slice`

但多模态 / 动态 batch 场景下，当前 step 实际参与计算的 `x` 往往只是一个收窄后的 token 视图。此时如果索引仍沿用完整批次长度，就会出现：

- `x` 的第一维
- `y` 的第一维
- `token_lora_indices` 的第一维

三者不一致。

新增 `_get_token_lora_indices(x)` 后，decode 路径在 kernel 调用前先用 `x.size(0)` 对 `_token_lora_indices` 做 `narrow`，这背后解决的不是某个 shape bug，而是一个更基础的问题：**元数据生命周期与实际计算窗口脱节。**

它的深层价值是：

1. 明确 decode 计算只应消费当前 step 的有效 token 视图  
   LoRA 索引不是全局静态背景信息，而是需要与当下参与运算的 token slice 同步演进的元数据。
2. 提升多模态、动态裁剪场景下的健壮性  
   这类场景最容易暴露“张量内容被裁剪了，但配套索引没裁剪”的问题。修复后 Punica NPU 路径对动态 batch 视图更鲁棒。
3. 为后续 kernel 级优化保留一致性前提  
   只有先保证输入与索引严格同长，后面的 kernel 替换、融合或算子下沉才有稳定基础。

所以这条修复的本质是：**把 LoRA decode 从“张量视图已变、元数据还停留在旧批次语义”的状态，修正为计算窗口与元数据严格一致。**

### 2.1.8 `model_runner_v1.py::_torch_cuda_wrapper()` 保留原始异常

这条修复的架构价值，是**恢复异常语义的透明传递**。

修复前，`_torch_cuda_wrapper()` 在内部异常后会统一重新包装为：

- `RuntimeError("NPUModelRunner init failed, error is ...")`

这会带来一个典型问题：upstream 很多测试、调用链、乃至业务分支其实依赖“异常类型本身”做断言或决策。插件一旦把所有异常压平为 `RuntimeError`，就等于把上游精心设计的失败语义全部抹掉了。

修复后改成：

- 记录 `logger.exception(...)`
- `raise` 原始异常

它的价值在于：

1. 插件不再重写 upstream 的错误协议  
   作为平台适配层，插件可以补充日志，但不应擅自改写异常类型语义。
2. 提高问题定位效率  
   原始异常类型和 traceback 会直接暴露真实失败点，避免排障时被一层泛化 `RuntimeError` 稀释掉上下文。
3. 让测试失败更有区分度  
   对 rejection 类用例尤其重要，因为这类测试验证的是“系统能否拒绝非法输入并给出正确类别的错误”，不是仅仅验证“有错误发生”。

这条修复的更深价值，可以概括成一句话：**平台插件应该增强可观测性，而不是篡改失败语义。**

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

如果只保留最关键的改动 -> 用例映射，可以记成下面 9 条：

1. `platform.py::check_and_update_config()` CPU/no-model 早返回
   - 直接去噪并恢复一批 `tests/v1/kv_connector/unit/*` upstream 逻辑测试。
2. `ascend_config.py` 的 `model_config` 显式保护
   - 直接修复 `tests/v1/kv_connector/unit/test_config.py`。
3. `platform.py::get_device_total_memory()` + `npu-smi` 路径
   - 直接修复 `tests/v1/engine/test_engine_args.py`。
4. `platform.py` 默认未指定时改用 `spawn`
   - 直接消除了 `test_qwenvl.py` 中由 `fork` 引起的 `Invalid thread pool!` 首错；当前已与用户手动整文件复跑结果共同收敛为 `6 passed`。
5. `lora/utils.py` 恢复 upstream 等价的 LoRA wrapper 分流
   - 直接修复 `tests/lora/test_add_lora.py`，并为 `tests/lora/test_qwenvl.py` 的稳定通过恢复了正确的 packed/merged 语义前提。
6. `punica_npu.py` 的 token-index 收窄
   - 直接消除了 `tests/lora/test_qwenvl.py` 旧的 LoRA decode 维度错配；与默认 `spawn` 修复共同构成当前整文件 `6 passed` 的代码前提。
7. `model_runner_v1.py::_torch_cuda_wrapper()` 保留原始异常类型
   - 直接修复 `test_custom_offline.py::test_rejects_custom_logitsprocs` 的 6 个 rejection 子例。
8. `ops/linear.py` 去掉对 `output_sizes` 的错误默认覆写
   - 不单独宣称修复某个用例，但它与 `lora/utils.py` 一起构成 LoRA 类型语义恢复的必要代码前提，避免用“伪造 packed 元信息”的方式掩盖 wrapper 选型问题。
