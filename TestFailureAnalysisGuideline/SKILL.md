# Test Failure Analysis Guidelines for vLLM-Ascend

The final goal is to identify a portfolio of upstream vLLM testcases that should be added to the vllm-ascend CI pipeline to improve functional guard coverage for Ascend, while preserving upstream behavior and logic rather than re-defining them.

## Scope

- The primary repositories are `vllm` and `vllm-ascend`. vLLM-Ascend is a platform plugin that adapts vLLM to run on Ascend hardware, so the adaptation boundary is a key focus for test selection and failure analysis.
- When asked to analyze a batch of failing tests, inspect both repositories if the failure path crosses the adaptation boundary.

## Required Context for Analysis

Each analysis should record:

- vllm commit / version
- vllm-ascend commit / version
- Python / torch / torch-npu / CANN versions
- whether the environment is clean or pre-cached
- whether external network is allowed


## Root-Cause-First Workflow

1. Run the target test(s) and capture the first real failure.
2. Do not treat network, download, cache-miss, or missing-dependency errors as final conclusions unless they remain the true blocker after reasonable mitigation.
3. If needed, eliminate environmental noise so the underlying failure can surface.
4. Read the relevant code paths before concluding whether the issue belongs to `vllm`, `vllm-ascend`, the test itself, or the runtime stack.

## UT precondition preparation

When analyzing a failure, you may perform the following mitigations to expose the real root cause:

- download model weight by ModelScope first(export VLLM_USE_MODELSCOPE=True), fallback to huggingface if model is unavailable on ModelScope
- install the missing Python dependencies, and record the exact versions installed
- install the required model weights if <= 7B parameters, otherwise classify as "external model too large for CI"
- if resources not listed here are required, ask me before installing or caching them
- if the failure is due to transient network or environment issues, like download speed too slow, ask me before retrying

Do not stop at "network issue" if a practical workaround can expose a deeper failure.

If exposing the real failure requires extensive setup or significant deviation from the original test intent, classify the test as unsuitable for CI instead of continuing mitigation.

## Environment Support

Use following ENV variables to access internet:
export http_proxy=socks5h://127.0.0.1:1080
export https_proxy=socks5h://127.0.0.1:1080

Use ModelScope to download model weight:
export VLLM_USE_MODELSCOPE=True

## Testcase Selection Guidelines

Candidate testcases should be selected using the following rules:

### 1. Selection Principles

A testcase is a strong CI candidate if it satisfies most of the following:

- it validates an upstream behavior contract that vllm-ascend is expected to preserve
- it exercises an Ascend-sensitive adaptation boundary, such as hardware plugin loading, platform dispatch, custom op registration, model adaptation, compiler/runtime fallback, or backend-specific execution paths
- it is stable and reproducible in a controlled CI environment
- it has high signal-to-noise ratio, meaning failures are likely to indicate a real code, adaptation, or runtime regression rather than transient environment issues
- it has manageable setup cost in CI, including model size, asset size, startup time, and dependency footprint
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
- it depends heavily on unstable external network access, unavailable models, or large remote assets
- it is primarily a benchmark, throughput check, or performance characterization rather than a correctness guard
- it is highly flaky due to timing, process startup ordering, port collisions, or nondeterministic distributed behavior
- it requires broad environment mutation beyond minimal targeted mitigation
- it tests a feature that is explicitly unsupported or not applicable on Ascend
- it duplicates coverage already provided by a smaller, more stable testcase
- it requires disproportionately high analysis or setup cost relative to its regression detection value

## Output Requirements

When summarizing a batch of failing tests:
- provide a table with each test, the true root cause
- clearly list any dependencies installed, resources cached, or workarounds used during analysis
- call out any test file listed by the task that does not actually exist in the checked-out branch
- state whether the testcase is worth adding coverage for in `vllm-ascend`, and why according to testcase selection guidelines. If not, explain the main blockers and suggest what would be needed to make it a good candidate.

When recommending testcases for CI admission, provide:
- what upstream behavior it is guarding and why that behavior is relevant to Ascend
- the ownership (which component should address it: vllm, vllm-ascend, test, or runtime stack)
- can the testcase be made to pass with code fixes in `vllm-ascend`, or does it require changes to upstream `vllm` or the test itself, or other mitigation beyond code changes?


## Project Summary

For `vllm-ascend`, upstream unit tests should be selected primarily from behavior-contract layers rather than CUDA-specific implementation layers. 

The goal is to ensure that `vllm-ascend` preserves upstream behavioral contracts, rather than reproducing backend-specific implementations.

Priority should be given to tests that guard hardware plugin loading, platform dispatch, CustomOp fallback, config normalization, API/request validation, scheduler and cache semantics, LoRA adapter loading and module mapping, lightweight multimodal input handling, pooling task contracts, and OpenAI-compatible API path correctness.
Lower priority should be given to tests that are backend-specific to CUDA/HIP, heavily network-dependent, benchmark-oriented, or operationally flaky.

Presubmit CI should contain the smallest stable subset with high regression signal, while heavier multimodal, runtime-compatibility, and distributed-setup cases should be deferred to nightly CI.

