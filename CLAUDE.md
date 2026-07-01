<!-- This file is duplicated as CLAUDE.md and AGENTS.md. Keep them in sync. -->
# AGENTS.md — Workspace-level instructions for AI coding agents

## DeepSpeed Project Rules

### Commit & CI requirements

- All commits MUST have a `Signed-off-by` line (use `--signoff`). Get the name and email from `git config user.name` / `git config user.email`.
- Formatting: yapf (column_limit=119, `.style.yapf`) + flake8 (`.flake8`).
- Always verify changed files pass pre-commit checks before committing: `pre-commit run --files <changed_files>`. Only check modified files, not the entire codebase. Config: `.pre-commit-config.yaml`.
- `check-torchdist` hook: NEVER directly import torch's distributed module. Use `import deepspeed.comm as dist` instead.
- New files require license header:
  ```
  # SPDX-License-Identifier: Apache-2.0
  # DeepSpeed Team
  ```

### Code change discipline

- NEVER make cosmetic/formatting-only changes to existing code. Only add/modify lines that are functionally necessary. Minimizing diff noise is critical for code review.
- Delete dead code decisively — if code is unused at runtime (only referenced in tests), remove it along with its tests.
- Prefer consolidating tests over proliferating test files.
- Blend in: when modifying code, read the surrounding context and match the style of neighboring code (naming, spacing, patterns, idioms).
- Write beginner-friendly code: avoid deeply nested expressions or chained logic. Break complex expressions into clear, named intermediate steps.
- Comments should explain **why**, not **what**. Describe the purpose and reasoning, not the mechanics that the code already shows.
- New features must include corresponding tests and documentation updates.

## Execution discipline (CRITICAL)

- **NO commentary.** Do not narrate what you are about to do, what you just did, or why. Execute silently.
- **NO progress reports.** Do not tell the user "I'm now going to...", "Let me first...", "Step 1:", "Here's my plan:". Just do it.
- **NO asking for confirmation.** If the task is clear, execute. Do not ask "shall I proceed?" or "would you like me to...".
- **NO summarizing your own output.** After running a command or writing code, do not restate what the output says. The user can read.
- **NO hedging.** Do not say "I think", "it seems like", "this might". State facts or act.
- **Maximize code, minimize prose.** A response with 200 lines of code and 0 lines of explanation is better than 20 lines of code and 50 lines of explanation.
- **Full files, not fragments.** When modifying code, output the complete modified file via artifacts or write it directly. Do not output partial diffs with "..." elisions.
- **Loop until done.** If a task requires multiple steps, execute all of them in one turn. Do not stop after one step and wait for the user to say "continue".
- **No new branches, no v2/v3 suffixes.** All work goes on main. All files keep their original names.

## Tool Caveats

### Edit tool auto-formatter

The Edit tool has a hidden auto-formatter that silently changes quotes, whitespace, blank lines, and line wrapping. For format-sensitive modifications (e.g., when exact formatting matters for pre-commit), use `bash` with `sed`, `python`, or `cat` instead.
