---
name: Test Failure Analysis Skill
description: Workflow and decision criteria for identifying, analyzing, and triaging failing tests in vllm-ascend.
---

# Test Failure Analysis Skill for vLLM-Ascend

The final goal is to identify a portfolio of upstream vLLM testcases that should be added to the vllm-ascend CI pipeline to improve functional guard coverage for Ascend, while preserving upstream behavior and logic rather than re-defining them.

## Scope

- The primary repositories are `vllm` and `vllm-ascend`. vLLM-Ascend is a platform plugin that adapts vLLM to run on Ascend hardware, so the adaptation boundary is a key focus for test selection and failure analysis.
- When asked to analyze a batch of failing tests, inspect both repositories if the failure path crosses the adaptation boundary.
- Primary workspace and codes are under:
    - `/vllm-workspace/vllm`
    - `/vllm-workspace/vllm-ascend`

## Required Context for Analysis

Each analysis should record:

- vllm commit / version
- vllm-ascend commit / version
- Python / torch / torch-npu / CANN versions


## Root-Cause-First Workflow

1. Run the target test(s) and capture the first real failure. Remember each test file or testcase should be run individually on an empty NPU card to avoid OOM and get the real failure!
2. Do not treat network, download, cache-miss, or missing-dependency errors as final conclusions unless they remain the true blocker after reasonable mitigation.
3. If needed, eliminate environmental noise so the underlying failure can surface.
4. Read the relevant code paths before concluding whether the issue belongs to `vllm`, `vllm-ascend`, the test itself, or the runtime stack.

## Environment Support & Rules

Assume environment has China mainland network access, e.g. curl https://hf-mirror.com should work.

Rule0: Do not stop at "network issue" or "missing dependency" if a practical workaround can expose a deeper failure!!!

Rule1: Each test should be run on an empty NPU card to avoid OOM and get the real failure. Use following ENV variable to select NPU cards:
Use following ENV variable to select NPU cards:
export ASCEND_RT_VISIBLE_DEVICES=6,7
And use 'npu-smi info' to check the available NPU cards, should use empty card for CI test to avoid OOM issue.

Rule2: Use huggingface mirror to download model weight and other required files:
export HF_ENDPOINT=https://hf-mirror.com

Rule3: pip install using aliyun mirror:
pip install -i https://mirrors.aliyun.com/pypi/simple/ <package-name>

## Testcase analysis for Ascend CI

Candidate testcases should be selected using the following rules:

### 1. Principles

A testcase is a strong CI candidate if it satisfies most of the following:

- it validates an upstream behavior contract that vllm-ascend is expected to preserve
- it exercises an Ascend-sensitive adaptation boundary, such as hardware plugin loading, platform dispatch, custom op registration, model adaptation, compiler/runtime fallback, or backend-specific execution paths
- it is stable and reproducible in a controlled CI environment
- it has high signal-to-noise ratio, meaning failures are likely to indicate a real code, adaptation, or runtime regression rather than transient environment issues
- it covers a previously observed or highly plausible regression mode in vllm-ascend
- it can run with local or pre-cached assets, or with minimal deterministic preparation
- it behaves deterministically under fixed inputs, seeds, and environment

### 2. Prioritization Rules

Prefer testcases that are:

- P0: core compatibility guards
Tests that verify `vllm-ascend` does not break core upstream semantics or public APIs
- P1: adaptation-boundary guards
Tests that validate integration points where upstream logic meets Ascend-specific adaptation
- P2: feature guards
Tests for supported Ascend features that have a history of regressions, such as LoRA, multimodal, pooling, or OpenAI-compatible serving
- P3: extended coverage
Tests that add useful confidence but are expensive, flaky, or lower-yield for every-commit CI

### 3. Exclusion Rules

A testcase is usually not a good CI candidate if:

- it mainly validates CUDA-only, HIP-only, or non-Ascend-specific backend behavior
- it tests a feature that is explicitly unsupported or not applicable on Ascend

## Output Requirements

When summarizing a batch of failing tests:
- provide a table with each test, the true root cause
- testcase analysis for each test(Granularity at the file level), including which upstream behavior it is guarding and why that behavior is relevant to Ascend, the ownership (which component should address it: vllm, vllm-ascend, test, or runtime stack), and whether the testcase can be made to pass with code fixes in `vllm-ascend` or requires changes to upstream `vllm` or the test itself, or other mitigation beyond code changes
- clearly list any dependencies installed, resources cached, or workarounds used for each test
- whether the testcase can be made to pass with code fixes in `vllm-ascend` or requires changes to upstream `vllm` or the test itself, or other mitigation beyond code changes


## Project Summary

For `vllm-ascend`, upstream unit tests should be selected primarily from behavior-contract layers rather than CUDA-specific implementation layers. 

The goal is to ensure that `vllm-ascend` preserves upstream behavioral contracts, rather than reproducing backend-specific implementations.

Priority should be given to tests that guard hardware plugin loading, platform dispatch, CustomOp fallback, config normalization, API/request validation, scheduler and cache semantics, LoRA adapter loading and module mapping, lightweight multimodal input handling, pooling task contracts, and OpenAI-compatible API path correctness.
Lower priority should be given to tests that are backend-specific to CUDA/HIP, heavily network-dependent, benchmark-oriented, or operationally flaky.

## Case Studies

- [case_lora_wrapper_selection_gap.md](case_lora_wrapper_selection_gap.md)
    - 典型插件适配边界案例：`vllm-ascend` 对 upstream LoRA wrapper 选择逻辑迁移不完整，导致 `test_add_lora.py` 在 `set_lora()` 阶段触发 `IndexError`。
    - 重点经验：当失败只在安装插件后出现，且 traceback 落在 upstream 层代码时，要优先检查插件是否遗漏了 upstream 的分流条件（如 `packed_modules_list`、`output_sizes`），而不是直接怀疑 upstream 本身。
