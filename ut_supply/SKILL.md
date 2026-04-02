---
name: Unit and Coverage Test Supply Skill
description: Guidelines for adding comprehensive unit tests in HuggingClaw modules to achieve line and branch coverage targets.
---

# Unit and Coverage Test Supply Skill

## Overview

This skill is for senior test engineers creating or improving automated tests to reach high code coverage while maintaining meaningful assertions and low risk.

## When to use

- When implementing tests for a target module in HuggingClaw
- When coverage is missing due to untested logic paths

## Steps

1. Analyze public APIs, key functions, control flow branches, and coverage gaps.
2. Plan scenarios for happy paths, invalid input, error paths, boundaries, config flags, and dependency failures.
3. Implement focused tests that cover all branches and assert behavior.
4. Validate by running tests and checking coverage, iterating to 100% line and branch coverage where feasible.

> Important: The target is to get as close as possible to 100% line and branch coverage. Do not stop midway; continuously iterate and improve until coverage is extremely close to 100%.

## Rules

- Prefer test additions over production changes.
- Avoid rewriting production code unless required for correctness/testability.
- Align with existing test style (framework, naming, mocking patterns).
- Ensure tests are meaningful and not coverage inflation.

## Quality requirements

- Use precise assertions.
- Cover empty/null/min/max values, error paths, defaults, and short-circuit logic.
- Control randomness and time to avoid flakiness.

## Output

- New or updated test files
- Short summary of scenarios added
- Final coverage report with line/branch results

## Notes

- If a path is hard to test, use mocks/dependency injection.
- Keep production code edits minimal and justified.


Start working once the target module is provided.
