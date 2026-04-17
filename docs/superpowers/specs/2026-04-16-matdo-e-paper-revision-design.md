# MATDO-E Paper Revision Design

## Objective

Revise the current `MATDO-E.tex` manuscript into a safer anonymous-submission draft that prioritizes defensibility over maximal rhetorical ambition, then produce:

- root-level English LaTeX manuscript: `matdo-e_paper.tex`
- root-level Chinese markdown manuscript: `matdo-e_paper_cn.md`

## Revision Mode

This revision follows a `submission-oriented` standard with `no venue binding`.

The target is a generally credible anonymous ML submission in the style of ICLR/NeurIPS/ICML, but without tailoring to a specific conference template or idiosyncratic venue norm. The writing should therefore optimize for:

- reviewer trust
- claim-evidence alignment
- clear scope boundaries
- stable terminology
- lower attack surface on novelty and empirical overreach

## Source And Deliverables

### Primary source

- `MATDO-E.tex`

### Final deliverables

- `matdo-e_paper.tex`
- `matdo-e_paper_cn.md`

The English manuscript is the source of truth. The Chinese manuscript should faithfully reflect the final English version and should not introduce stronger claims, additional evidence, or new citations.

## Core Repositioning

The current draft reads too much like a broad programmatic declaration. The revised draft should instead present MATDO-E as:

- a unified analytical framework over four inference-time resource controls `(R, M, T, E)`
- a set of structural results derived under an explicit modeling approximation
- two implementation-level enhancements that fit within that framework
- an empirical study showing supportive trends and favorable trade-offs in the stated evaluation setting

The revised paper should **not** read as if it definitively solves the LLM memory wall or universally dominates all baselines in all settings.

## Main Revision Principles

### 1. Lower unsupported or overbroad claims

Systematically weaken formulations that imply more than the draft can safely support. Typical rewrites include:

- `the first principled foundation` -> `a unified analytical framework`
- `necessary and sufficient` -> `under the stated assumptions` or `admits an iff characterization in our model`
- `comprehensive validation` -> `empirical validation across three model families`
- `significantly outperforming state-of-the-art systems` -> `showing favorable trade-offs against representative baselines in our evaluation setting`

### 2. Preserve real contributions while changing emphasis

Contribution priority should be:

1. the unified `(R, M, T, E)` resource abstraction
2. structural phenomena predicted by the model, including dual walls and near-wall divergence
3. algorithmic enhancements: `msign-informed qTTT` and `information quality awareness`
4. empirical trade-off improvements under the reported setup

This avoids presenting the paper as a loose combination of multiple unrelated ideas.

### 3. Make assumptions explicit in theory sections

Theory sections should clearly distinguish:

- what is modeled
- what is approximated
- what is proved within the model
- how the result should be interpreted

In particular, the draft should state modeling assumptions around:

- additive error decomposition
- Engram compensation function
- retrieval cost model
- budget coupling simplifications

### 4. Reframe experiments around validation questions

The experiments should answer targeted questions instead of reading like a generic leaderboard comparison. The experimental narrative should support three questions:

- does the unified model predict the observed near-wall trend?
- does adding `E` shift the feasible operating region?
- do the proposed enhancements improve the trade-off inside the same framework?

## Section-Level Rewrite Plan

### Title

Keep the `memory wall + unified framework` framing, but reduce slogan-like or manifesto-like tone.

### Abstract

Rewrite into a stable 5-part structure:

1. problem and gap
2. proposed framework
3. theoretical findings
4. method-level enhancements
5. empirical findings within the stated evaluation setting

### Introduction

Strengthen the logic chain:

- what concrete deployment tension defines the memory wall
- why current methods remain fragmented
- why `(R, M, T, E)` is a useful abstraction
- what question the paper answers
- where the contribution boundary is

### Contributions

Reduce to 3-4 paper-style contributions. Each item should be independently legible and reviewer-checkable.

### Related Work

Reorganize by comparison axis rather than by citation list. Each subsection should end with a positioning statement that explains how MATDO-E differs from prior work.

### Model / Theory

Retain the current mathematical backbone, but add:

- clearer setup prose
- assumption-aware interpretation
- more careful phrasing around closed-form claims
- narrower extrapolation from the model to real systems

### Experiments

Reframe the section from `we win everywhere` to `we validate targeted predictions and compare trade-offs under the same serving setting`.

### Conclusion / Limitations

End with a measured summary:

- what the framework contributes
- what depends on simplifying assumptions
- what appears setting-dependent in experiments
- what remains open for future work

## English Output Contract

`matdo-e_paper.tex` should:

- remain anonymous
- preserve the central mathematical structure unless a claim needs softening
- improve abstract, introduction, related work, contribution framing, experiments narrative, and limitations
- maintain internal terminology consistency
- avoid inventing new experimental results, citations, or theoretical guarantees

## Chinese Output Contract

`matdo-e_paper_cn.md` should:

- be a faithful Chinese academic rendering of the final English paper
- avoid line-by-line translation stiffness
- preserve claim boundaries from the English version
- use stable term mappings, including:
  - `memory wall` -> `内存墙`
  - `resource knobs` -> `资源控制变量` or `资源旋钮`
  - `context wall` -> `上下文墙`
  - `compute wall` -> `计算墙`
  - `heterogeneous resource arbitrage` -> `异构资源套利`
  - `information quality awareness` -> `信息质量感知`

## Explicit Non-Goals

This revision will not:

- fabricate new experiments
- fabricate new citations
- upgrade weak evidence into strong claims
- rewrite the project into a venue-specific format
- turn the Chinese version into a promotional article

## Risks To Watch

- theory remains stronger in tone than the assumptions justify
- experiment tables are interpreted more broadly than the setup supports
- related work still sounds dismissive toward prior work
- the two algorithmic enhancements overshadow the unified-framework story
- English and Chinese versions drift in claim strength

## Success Criteria

The design is successful if the revised paper:

- reads like a coherent anonymous submission rather than a project summary
- has lower reviewer attack surface on claim strength
- presents a cleaner contribution hierarchy
- keeps theory, experiments, and limitations aligned
- yields two consistent final deliverables: `matdo-e_paper.tex` and `matdo-e_paper_cn.md`

## Execution Order

1. revise the English manuscript into `matdo-e_paper.tex`
2. derive `matdo-e_paper_cn.md` from the finalized English manuscript
3. run a consistency pass across title, abstract, contributions, experiments, and limitations