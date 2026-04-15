---
name: PR Guidelines Skill
description: Standardize pull request conventions for vLLM-Ascend
---

# PR Guidelines Skill

This directory records local pull request conventions that should be followed when preparing or updating PRs for repositories in this workspace.

## vllm-ascend

### PR title prefixes

PR titles must contain one of the following exact prefixes:

- `[BugFix]`
- `[Performance]`
- `[Test]`
- `[CI]`
- `[Feature]`
- `[Doc]`
- `[Misc]`
- `[Community]`
- `[Refactor]`

Examples:

- `[BugFix] Fix Ascend config boundary handling`
- `[Feature] Add new optimization pass`

### Current preference

For bug-fix PRs in `vllm-ascend`, prefer `[BugFix]` and keep the rest of the title concise and behavior-focused.

### Commit and DCO requirements

- All commits in a PR must satisfy DCO.
- Create commits with `git commit --signoff`.
- When amending, keep the signoff with `git commit --amend --signoff`.
- If a PR already contains unsigned commits, rewrite them before merging. Do not assume one signed follow-up commit is enough.

### Lint and formatting requirements

- Treat `pre-commit` as the source of truth for lint and formatting.
- Before pushing, prefer running `pre-commit run --all-files`.
- Pay special attention to `ruff check` and `ruff format`; CI may fail even for small style issues.
- Keep Python lines within the configured width limit and prefer `ruff`-friendly multiline formatting for long boolean expressions.
- If CI shows "files were modified by this hook", apply the hook-produced diff locally and recommit instead of trying to outguess the formatter.

### PR body guidance

- Frame `vllm-ascend` as an out-of-tree platform plugin.
- Emphasize preserving upstream vLLM behavior contracts rather than redefining them.
- Prefer behavior-level summaries over file-by-file inventories.
- PR descriptions must follow the structure of `references/PULL_REQUEST_TEMPLATE.md`.
- When preparing a PR body, use the sections in `references/PULL_REQUEST_TEMPLATE.md` as the baseline template.
- When relevant, connect fixes to recovered upstream tests, but keep the evidence list short.
