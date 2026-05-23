# Gate21 Official SeHGNN Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gate21 SeHGNN placeholder with a real official SeHGNN-model bridge that trains from Gate21 exports and saves validation/test logits.

**Architecture:** Keep the existing Gate21 export/audit/summarizer surfaces. Add relation schema metadata to exports, add a subprocess runner that loads the official `external/SeHGNN/hgb/model.py` model class, trains with preserved Gate21 splits, writes logits and metrics, then have `sehgnn_bridge.py` parse the runner result.

**Tech Stack:** Python, NumPy, PyTorch, official SeHGNN model module, pytest, existing Gate21 CLI.

---

### Task 1: Preserve Relation Schema in Exports

**Files:**
- Modify: `hesf_coarsen/eval/official/graph_export.py`
- Test: `tests/eval_official/test_graph_export.py`

- [ ] Add a failing assertion that `metadata.json` includes `relation_schemas` with relation name, source type, and destination type.
- [ ] Add `relation_schemas` to the export metadata using `graph.relation_specs`.
- [ ] Run `conda run -n pytorch python -m pytest tests/eval_official/test_graph_export.py -q`.

### Task 2: Build SeHGNN Target Feature Blocks

**Files:**
- Create: `hesf_coarsen/eval/official/sehgnn_export_runner.py`
- Test: `tests/eval_official/test_sehgnn_bridge.py`

- [ ] Add a failing unit test that exports a tiny graph and verifies target-self plus neighbor-aggregate feature blocks are created.
- [ ] Implement deterministic dense mean aggregation for relations touching the target type.
- [ ] Run `conda run -n pytorch python -m pytest tests/eval_official/test_sehgnn_bridge.py -q`.

### Task 3: Train Official SeHGNN in a Subprocess

**Files:**
- Create: `hesf_coarsen/eval/official/sehgnn_export_runner.py`
- Modify: `hesf_coarsen/eval/official/sehgnn_bridge.py`

- [ ] Implement runner CLI accepting `--export-dir`, `--repo-dir`, `--dataset-name`, `--target-type`, `--seed`, and `--result-json`.
- [ ] Dynamically load official SeHGNN `hgb/model.py`, train with Gate21 train/val/test splits, save `val_logits.npy` and `test_logits.npy`.
- [ ] Update `run_sehgnn_official` to execute the runner with `subprocess.run`, capture stdout/stderr, and return unified metrics.
- [ ] Run a DBLP full-graph single-run smoke through `run_gate21_open_sota_bridge.py`.

### Task 4: Run Gate21 Matrix and Verify

**Files:**
- Outputs under `outputs/gate21_open_sota/`

- [ ] Run export-only DBLP full and H6.
- [ ] Run DBLP seed 23456 full and H6 SeHGNN official with calibration.
- [ ] Run the minimal Gate21 matrix for SeHGNN official.
- [ ] Run OpenHGNN-SeHGNN smoke after SeHGNN is stable and record clear status.
- [ ] Run `conda run -n pytorch python -m pytest tests/eval_official -q`.
- [ ] Re-read `gate21_requirement_checklist.md` and report every checked/unchecked item.
