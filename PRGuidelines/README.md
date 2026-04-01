# PR Guidelines

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

### PR body guidance

- Frame `vllm-ascend` as an out-of-tree platform plugin.
- Emphasize preserving upstream vLLM behavior contracts rather than redefining them.
- Prefer behavior-level summaries over file-by-file inventories.
- When relevant, connect fixes to recovered upstream tests, but keep the evidence list short.
