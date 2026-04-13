# vllm-ascend `core/` UT 覆盖率提升案例

本文整理一次真实的 UT 覆盖率提升过程，目标是给后续 agent 提供一个**可直接复用**的参考案例。

目标模块：

- `vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py`
- `vllm-ascend/vllm_ascend/core/recompute_scheduler.py`

目标：

- 在**不修改生产代码**的前提下，尽可能把 line / branch coverage 拉到接近 100%
- 输出可长期维护、可回归验证的 UT

---

## 一、最终结果

本次补充完成后，focused core UT 的结果为：

- `92 passed, 5 warnings`

最终覆盖率：

- `scheduler_dynamic_batch.py`
	- line / statement: `269/273 = 98.53%`
	- branch: `111/118 = 94.07%`
- `recompute_scheduler.py`
	- line / statement: `459/467 = 98.29%`
	- branch: `212/232 = 91.38%`

这已经可以视为“非常接近 100%”的实战结果。

---

## 二、为什么这次案例值得参考

这两个文件并不是简单的纯函数模块，而是典型的**重状态、重调度、依赖复杂**的工程代码：

- 依赖大量 `vllm` 内部对象和配置
- 有等待队列 / 运行队列 / preempt / resumed / remote KV 等状态机分支
- 有 multimodal、LoRA、spec decode、KV connector、EC connector、stats 等横切逻辑
- 既有 `schedule()` 大函数，也有 `update_from_output()` 这种后处理大函数

这种模块最容易出现两类错误：

1. 只补 happy path，coverage 上不去
2. 为了追 coverage 去改生产代码，导致风险扩大

本案例的关键点就是：**只补测试，不碰生产逻辑，用高收益分支测试把 coverage 拉上去。**

---

## 三、采用的方法论

### 1. 先分层，不要一上来就写“大而全”测试

这次有效的方法是把测试拆成三层：

#### 第一层：低成本 helper / config 分支

先覆盖稳定、小型、低耦合逻辑，例如：

- `BudgetRefiner`
- `RecomputeSchedulerConfig.initialize_from_config`
- `RecomputeScheduler.__init__`
- `add_request`
- `_update_waiting_for_remote_kv`

收益：

- 很快建立基础覆盖
- 帮助理解对象状态和依赖关系
- 形成后续 integration-style 用例需要的夹具认知

#### 第二层：真实 scheduler fixture + 主流程测试

为 `schedule()` / `update_from_output()` 搭可运行的真实 fixture：

- `SchedulerConfig`
- `ModelConfig`
- `VllmConfig`
- `CacheConfig`
- `DeviceConfig("cpu")`
- `KVCacheConfig`
- fake multimodal registry

然后先补：

- 基础 schedule
- multimodal schedule
- stop via output
- spec decoding stats
- memory leak / cleanup

#### 第三层：coverage 驱动的定点补洞

真正把 coverage 拉高的关键，不是继续加泛化集成测试，而是：

- 看 coverage report
- 找未覆盖 line / branch cluster
- 为每一簇分支补 1~2 个最小场景测试

这一步收益最高。

---

## 四、本次实际踩坑与解决办法

### 1. `pytest` 默认链路太重，先避开 `conftest`

直接跑默认 pytest 时，`tests/ut/conftest.py` 带来的 import / patch 链路很重，会导致：

- 启动慢
- 插件链复杂
- 失败定位困难

本次稳定做法：

```bash
VLLM_PLUGINS='' python -m pytest --noconftest -q \
	tests/ut/core/test_scheduler_dynamic_batch.py \
	tests/ut/core/test_budget_refiner.py \
	tests/ut/core/test_recompute_scheduler.py
```

建议：

- 对局部覆盖率提升任务，优先使用 `--noconftest`
- 对大仓库先做 focused run，而不是全量回归

### 2. 旧测试常常跟不上上游 API 演进

这次 `test_scheduler_dynamic_batch.py` 原有测试已经与当前 `vllm` API 存在漂移，主要体现在：

- `SchedulerConfig` 新参数要求变化
- `ModelConfig` 初始化参数变化
- `Request` 初始化参数变化
- `FullAttentionSpec` 构造方式变化
- multimodal dummy 输入接口变化

结论：

- 先让老测试重新跑起来，价值非常高
- 不要默认“已有测试可直接复用”

### 3. 对大函数要按“分支簇”而不是按“功能模块”补 UT

例如 `schedule()` 里最适合拆成这些分支簇：

- running queue 分支
- waiting queue gate 分支
- connector / remote KV 分支
- LoRA 限制分支
- encoder 输入分支
- preemption / recompute 分支
- event / stats / metadata 尾部分支

比起“写一个巨大场景覆盖所有逻辑”，这种方式：

- 更稳定
- 更容易定位失败
- 更容易继续迭代补洞

---

## 五、本次新增 / 修复的测试类型

### `test_budget_refiner.py`

主要覆盖：

- disabled path
- lookup table 缺失
- lookup table 读取与过滤
- key alignment
- table miss fallback
- exact lookup 命中
- refine budget 的 decode / non-decode 分支

### `test_scheduler_dynamic_batch.py`

主要覆盖：

- 基础调度
- multimodal 调度
- prefix caching
- stop via EOS / stop token / max tokens / ignore EOS
- concurrent batches
- spec decoding stats
- memory leak
- running request `num_new_tokens == 0`
- waiting remote KV ready / not ready
- waiting FSM ready / not ready
- async KV load
- allocation failure
- preempt / resumed request
- LoRA saturation
- max running seq cap
- long prefill threshold
- encoder budget 归零
- connector match unknown
- KV cache events + connector events 合并发布

### `test_recompute_scheduler.py`

主要覆盖：

- scheduler config 初始化
- `__init__` runtime flags
- `add_request` 的 streaming / placeholder / duplicate 分支
- `_update_waiting_for_remote_kv` 成功 / 失败 / partial / decrement 分支
- 基础 schedule
- multimodal schedule
- stop / stop token / length capped
- spec decoding stats
- recompute path
- remote KV ready / not ready
- FSM ready / not ready
- streaming waiting skip
- async KV load
- LoRA saturation
- running / waiting 的 long prefill threshold
- MTP request 的 spec clear / trim 分支
- encoder schedule zero 分支
- allocation failure + encoder free
- priority preemption rollback
- `use_v2_model_runner`
- EC metadata
- KV stats aggregation
- pooling / logprobs / NaN / finished request fanout
- recomputed output emission
- no visible output 分支
- cache + connector events merge

---

## 六、如何设计“高收益”测试

下面是这次证明很有效的几个套路。

### 套路 A：优先打等待态 gate

最容易漏 coverage 的，往往不是 happy path，而是 waiting queue 的 gate：

- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_FSM`
- `WAITING_FOR_STREAMING_REQ`
- LoRA saturation
- connector match unknown

这类测试通常只需要：

- 构造 1~3 个 request
- 手动设置 status
- mock 一个 helper 返回值
- 断言 request 被 skip / resume / schedule

收益很高。

### 套路 B：把 `schedule()` 尾部单独当成一簇

大函数尾部常常还有很多 coverage：

- build connector metadata
- take events
- merge events
- publish events
- attach stats
- finished request fanout

这些分支不需要真实大场景，通常只需要：

- 构造最小 `scheduler_output`
- mock connector / cache manager / publisher
- 断言 metadata / stats / events 是否出现

### 套路 C：不要怕“手动推进状态”

对 scheduler 类代码，很多高收益测试并不需要完整经过所有前序步骤。

完全可以直接设置：

- `request.status`
- `request.num_computed_tokens`
- `request.spec_token_ids`
- `request.num_output_placeholders`
- `scheduler.running`
- `scheduler.waiting`
- `scheduler.requests`

只要断言关注的是**行为结果**，这种做法完全合理。

### 套路 D：分支打不动时，先判断是不是“低 ROI / 非真实分支”

本次有一个典型例子：尝试覆盖 dynamic batch 中 running request 的“spec delta 非正且不记录”路径，但从当前实现约束看，这个路径很难在真实状态下稳定命中。

处理原则：

- 不要为了这类低收益路径把测试搞得非常脆弱
- 优先继续打剩余真实且稳定的分支

---

## 七、推荐执行顺序

后续 agent 如果再做类似任务，建议按下面顺序执行：

### Step 1：读目标模块 + 现有测试

- 先列出 public API / helper / 大函数
- 标出 if/else、continue/break、异常分支、状态切换点

### Step 2：修旧测试，让基础集成路径先跑通

- 如果现有测试已经漂移，先修它
- 这是后续 coverage 迭代的基础

### Step 3：补 helper / config / 状态转换的小测试

- 这类测试最快出成果

### Step 4：跑 focused coverage，不要上来全量

推荐命令：

```bash
VLLM_PLUGINS='' python -m coverage erase
VLLM_PLUGINS='' python -m coverage run --branch -m pytest --noconftest -q \
	tests/ut/core/test_scheduler_dynamic_batch.py \
	tests/ut/core/test_budget_refiner.py \
	tests/ut/core/test_recompute_scheduler.py

python -m coverage json -o coverage_focus.json \
	--include='*/vllm_ascend/core/scheduler_dynamic_batch.py,*/vllm_ascend/core/recompute_scheduler.py'
```

### Step 5：按 coverage report 定点补分支

重点看：

- missing lines
- missing branches
- 哪几段 line 连在一起

通常一簇相邻分支可以被 1 个测试打掉。

### Step 6：每轮只补少量高收益用例

建议每轮补：

- 3~8 个测试
- 跑 focused suite
- 再看 coverage

这样最稳。

---

## 八、可直接复用的判断规则

### 什么时候应该 mock

优先 mock：

- connector
- cache manager 边缘返回值
- stats / publisher
- grammar / structured output
- encoder scheduling helper
- external metadata builder

### 什么时候应该用真实 fixture

优先用真实 fixture：

- scheduler 主对象
- request 创建
- schedule / update_from_output 主流程
- multimodal / spec decode 这种对内部状态依赖较深的路径

### 什么时候不该改生产代码

如果仅仅是：

- 缺少某个边界分支测试
- 某个 helper 返回难构造
- 某个对象初始化太重

应优先：

- mock
- 手动塞状态
- 建更轻量的测试 helper

而不是修改实现。

---

## 九、给后续 agent 的模板

可以直接按这个模板开始：

1. 确认目标文件
2. 找现有测试
3. 先跑 focused pytest
4. 修漂移测试
5. 补 helper / config / state gate UT
6. 跑 focused coverage
7. 把 missing branches 按簇分类
8. 每轮补 3~8 个高收益测试
9. 直到 line / branch 达到目标区间

一句话总结：

> 大型调度器类的覆盖率提升，最有效的方法不是写一个超大集成测试，而是“真实 fixture + coverage 驱动的定点小测试”。

---

## 十、这份案例适用的模块类型

本案例特别适合以下类型的代码：

- scheduler / orchestrator
- queue / state machine
- pipeline controller
- engine output post-processor
- 带 connector / cache / stats / feature flag 的中枢模块

如果是纯函数模块，这份方法会偏重；但对工程化大类代码，这套方法非常有效。

