## HuggingClaw

HuggingClaw is a collection of practical skills and reference materials for common `vllm-ascend` engineering workflows. It focuses on a few high-frequency areas:

- test failure analysis and CI testcase selection
- unit test expansion and coverage improvement
- pull request conventions and pre-push checks
- case-study documentation for performance optimization

The repository contains both reusable `SKILL.md` workflow guides and concrete case-study documents. It is intended to be a practical reference when analyzing failures, adding tests, preparing PRs, or reviewing performance work.

## Directory Guide

### `pr-guidelines/`

Pull request guidance for `vllm-ascend`, including:

- PR title prefix conventions
- DCO / `--signoff` requirements
- lint and format expectations such as `pre-commit`, `ruff check`, and `ruff format`
- recommended PR body style

Use this when preparing or updating a `vllm-ascend` PR and you want to quickly confirm that the title, commit history, and description match local expectations.

### `test-failure-analysis-guideline/`

Failure-analysis guidance focused on diagnosing why upstream `vllm` tests fail on Ascend NPUs and deciding whether those tests are good CI candidates for `vllm-ascend`.

Core topics include:

- a root-cause-first debugging workflow
- environment requirements for running tests on Ascend NPUs
- criteria for deciding whether a testcase belongs in `vllm-ascend` CI
- ownership analysis across `vllm`, `vllm-ascend`, the test itself, and the runtime stack

The `references/` directory contains supporting analysis materials such as:

- a consolidated analysis of 128 tests
- mapping documents from tests to fixes
- focused failure analysis for specific test batches
- a representative LoRA wrapper selection gap case

Use this when triaging failing tests in batches, deciding whether an upstream testcase should be adapted into `vllm-ascend`, or selecting high-value functional guards for CI.

### `ut-coverage-improvement/`

Unit-test and coverage-improvement guidance aimed at pushing a target module toward very high line and branch coverage. The guidance emphasizes:

- preferring test additions over production-code changes
- systematically enumerating branches, boundaries, error paths, and dependency failures
- improving coverage with meaningful assertions rather than artificial execution

Its `references/` directory contains a real case study on improving unit-test coverage for a core `vllm-ascend` module.

Use this when expanding tests for an existing module, closing branch gaps, or preparing for review and CI coverage requirements.

### `cpu-binding/`

This is currently a standalone case-study directory containing a detailed write-up on `vLLM-Ascend` CPU binding optimization.

The case study focuses on:

- why CPU binding can reduce performance in an 8-card TP inference setup
- timing issues around `taskset` and `migratepages`
- the interaction between AutoNUMA, fork-shared pages, and steady-state hot pages
- how deferring CPU binding improves NUMA locality and stable performance

Use this as a reference when investigating CPU binding, NUMA locality, memory placement, and steady-state performance behavior in `vllm-ascend`.

## Recommended Entry Points

Choose an entry point based on the task:

- **Need to analyze failing tests?** Start with `test-failure-analysis-guideline/SKILL.md`
- **Need to add UTs or improve coverage?** Start with `ut-coverage-improvement/SKILL.md`
- **Need to prepare a PR?** Start with `pr-guidelines/SKILL.md`
- **Need a real case study?** Check the `references/` directories or standalone case-study documents

If you are working with an AI agent or Copilot-style workflow, it is usually helpful to include the relevant `SKILL.md` as part of the task context. That tends to reduce repeated trial-and-error significantly.

## Current Layout Style

Skill-oriented directories in this repository follow a consistent structure:

- keep `SKILL.md` at the directory root as the primary guide
- store case studies, analysis notes, and mapping documents under `references/`

`cpu-binding/` is currently still a case-study directory and has not yet been converted into the full `SKILL.md + references/` layout.

## Intended Audience

HuggingClaw is especially useful for:

- `vllm-ascend` developers
- engineers debugging upstream `vllm` behavior on Ascend NPUs
- contributors working on test expansion, failure triage, CI selection, and PR preparation

## Possible Next Steps

This repository can be extended with:

- more representative failure cases and fix mappings
- more fine-grained performance optimization topics
- a full skill-style conversion of `cpu-binding/` with `SKILL.md` and `references/`
- separate skill directories for multi-card, serving, quantization, and related topics
