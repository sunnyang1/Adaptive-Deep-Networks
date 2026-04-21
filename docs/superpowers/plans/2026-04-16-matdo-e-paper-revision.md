# MATDO-E Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the MATDO-E manuscript into a safer anonymous submission draft and produce aligned English and Chinese deliverables.

**Architecture:** Use `MATDO-E.tex` as the primary source, preserve the mathematical backbone, and rewrite the high-risk rhetorical and narrative sections first. Then generate a Chinese markdown version from the finalized English manuscript and run a consistency and lint pass.

**Tech Stack:** LaTeX, Markdown, repository-local manuscript sources

---

### Task 1: Lock the English manuscript structure

**Files:**
- Create: `matdo-e_paper.tex`
- Read: `MATDO-E.tex`
- Read: `docs/papers/matdo-e_paper.tex`
- Read: `docs/papers/matdo-e_revised_paper.md`

- [ ] Extract the reusable mathematical and experimental content from `MATDO-E.tex`.
- [ ] Rewrite title, abstract, introduction, contributions, related work, and conclusion to match the approved submission-oriented design.
- [ ] Keep formulas and tables only where they remain supportable under softened claim language.

### Task 2: Produce the Chinese manuscript

**Files:**
- Create: `matdo-e_paper_cn.md`
- Read: `matdo-e_paper.tex`

- [ ] Render a faithful Chinese academic version of the final English manuscript.
- [ ] Keep terminology stable and preserve the same claim boundaries.
- [ ] Prefer readable academic Chinese over literal sentence-by-sentence translation.

### Task 3: Verify alignment and quality

**Files:**
- Read: `matdo-e_paper.tex`
- Read: `matdo-e_paper_cn.md`

- [ ] Check title, abstract, contributions, experiments, and limitations for cross-language consistency.
- [ ] Run lint diagnostics on the edited files.
- [ ] Fix any easy formatting or syntax issues introduced during revision.
