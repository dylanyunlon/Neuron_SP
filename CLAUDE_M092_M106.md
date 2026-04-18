# CLAUDE_M092_M106.md — Handoff to Next Claude (M122+)

## What This Commit Series Does

16 commits total:
- 1 cleanup commit: -3968 lines (delete all Claude-5/7 appended virtual code)
- 15 critical merge commits (M092-M106): +102 lines across 15 files

## MANDATORY RULES FOR NEXT CLAUDE

1. **CRITICAL MERGE ONLY** — 2-10 lines at correct injection point per task
2. **NO new standalone classes appended at EOF**
3. **cat FILE FIRST, ast.parse AFTER**
4. **Wire into self.desloc_enabled / self.desloc_Kx** from engine.py
5. **MoE capacity allreduces must NOT be gated by Kx**
6. **ZERO numpy.random**
7. **Kx=1 must degrade to original behavior**
8. **DELETE virtual code if you find any**
