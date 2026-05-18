# Next17 Accuracy Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-invasive HeSF-Acc branch that tests target-preserve, target-selection/support-coarsening, full-target inference, task-aligned ablations, type budgets, and model-fidelity tagging.

**Architecture:** Keep the existing P/S preservation-first configs untouched. Build target-preserving and target-anchor hybrids from existing all-type assignments, evaluate them through clearly tagged Mode A and Mode B local SeHGNN/HETTREE protocols, and summarize each priority in separate output blocks.

**Tech Stack:** Python, NumPy, existing `HeteroGraph` / `Assignment` / `coarsen_graph`, local conda env `pytorch`, pytest.

---

### Task 1: Target-Preserve Assignment

**Files:**
- Create: `hesf_coarsen/coarsen/target_preserve.py`
- Test: `tests/test_target_nodes_not_merged.py`
- Test: `tests/test_support_only_coarsening.py`

- [ ] **Step 1: Write failing tests** asserting target nodes are singleton, support nodes can be coarsened, relation schema is preserved, and actual ratio is reported honestly.
- [ ] **Step 2: Implement `build_target_preserving_assignment`** using a base original-to-coarse mapping for support nodes while remapping every target node to a unique dense supernode.
- [ ] **Step 3: Run tests** with `conda run -n pytorch python -m pytest tests/test_target_nodes_not_merged.py tests/test_support_only_coarsening.py -q`.

### Task 2: Target Selection And Budgets

**Files:**
- Create: `hesf_coarsen/accuracy/target_selection.py`
- Create: `hesf_coarsen/accuracy/target_anchor_budget.py`
- Create: `hesf_coarsen/accuracy/type_budgets.py`
- Test: `tests/test_target_selection_budget.py`
- Test: `tests/test_target_selection_coverage.py`
- Test: `tests/test_type_budgets.py`

- [ ] **Step 1: Write failing tests** for mandatory train-target preservation, deterministic greedy anchors, per-class quota, and per-type budget reporting.
- [ ] **Step 2: Implement deterministic greedy selection** with confidence, uncertainty/margin, support coverage, diversity, and class-balance terms.
- [ ] **Step 3: Implement budget reporters** for target keep-all/select and support ratios.

### Task 3: Full-Target Inference Protocol

**Files:**
- Create: `hesf_coarsen/accuracy/full_target_inference.py`
- Test: `tests/test_full_target_inference.py`

- [ ] **Step 1: Write failing tests** for Mode A vs Mode B protocol tags.
- [ ] **Step 2: Implement wrappers** around `sehgnn_lite` and `hettree_lite` using official split overrides on the hybrid graph.
- [ ] **Step 3: Ensure summaries write separate transfer/full-target CSV files.**

### Task 4: Task-Aligned Accuracy Terms

**Files:**
- Create: `hesf_coarsen/accuracy/meta_recon.py`
- Create: `hesf_coarsen/accuracy/distillation.py`
- Create: `hesf_coarsen/accuracy/task_aligned_score.py`

- [ ] **Step 1: Implement lightweight meta/tree reconstruction diagnostics** for target-centered path features.
- [ ] **Step 2: Implement deterministic teacher proxy logits** and KL helper for distillation-aware rows.
- [ ] **Step 3: Mark A4/A5 ablations explicitly in output summaries.**

### Task 5: Next17 Runner And Summaries

**Files:**
- Create: `configs/accuracy/hgb_target_preserve_support_coarsen.yaml`
- Create: `configs/accuracy/hgb_hesf_acc_hybridA_keepall.yaml`
- Create: `configs/accuracy/hgb_hesf_acc_hybridB_selecttarget.yaml`
- Create: `configs/accuracy/hgb_hesf_acc_hybridB_selecttarget_distill.yaml`
- Create: `experiments/scripts/run_next17_hybrid_accuracy.py`
- Create: `experiments/scripts/summarize_next17_hybrid_accuracy.py`

- [ ] **Step 1: Build variants A0-A5** from existing Next15 all-type runs and generated hybrid graphs.
- [ ] **Step 2: Emit separate block outputs** for `target_preserve`, `hybrid`, `full_target_protocol`, `task_aligned`, `type_budget`, and `model_fidelity`.
- [ ] **Step 3: Run the bounded matrix** on ACM/DBLP/IMDB, ratios 2.4/4.8/9.6, at least 3 seeds.

### Task 6: Verification And Commit

- [ ] **Step 1: Run focused pytest.**
- [ ] **Step 2: Run Next17 local experiment with `conda run -n pytorch`.**
- [ ] **Step 3: If OOM occurs, write `server_commands.json`.**
- [ ] **Step 4: Commit all code changes to `main`.**
