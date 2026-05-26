# Gate21.6 Prompt Completion Checklist

- [x] Protocol A and Protocol B represented separately
- [x] All non-negotiable eligibility flags emitted in result rows
- [x] Structural/raw/cache/adapter ratios kept as separate columns
- [x] Graph-seed stability fields emitted
- [x] Safe feature ablation shape audit emitted
- [x] External TP baselines include Random/Herding/KCenter/Coarsening/GraphSparsify plus FreeHGC failure rows
- [x] Storage-only baselines emitted
- [x] System resource accounting emitted
- [x] Decision keeps FreeHGC/HGCond dependency gaps explicit

## Required Modules
- [x] `hesf_coarsen/eval/official/gate21_6_decision.py`
- [x] `hesf_coarsen/eval/official/icde_protocol.py`
- [x] `hesf_coarsen/eval/official/external_baselines_tp.py`
- [x] `hesf_coarsen/eval/official/freehgc_tp_adapter.py`
- [x] `hesf_coarsen/eval/official/coreset_tp_baselines.py`
- [x] `hesf_coarsen/eval/official/coarsening_tp_baseline.py`
- [x] `hesf_coarsen/eval/official/graph_sparsification_baselines.py`
- [x] `hesf_coarsen/eval/official/storage_only_baselines.py`
- [x] `hesf_coarsen/eval/official/adapter_package_manifest.py`
- [x] `hesf_coarsen/eval/official/safe_feature_transforms.py`
- [x] `hesf_coarsen/eval/official/metapath_cache_introspection.py`
- [x] `hesf_coarsen/eval/official/coverage_diagnostics.py`
- [x] `hesf_coarsen/eval/official/system_resource_logger.py`
- [x] `hesf_coarsen/eval/official/auto_relation_channel_selector.py`

## Required Scripts
- [x] `experiments/scripts/run_gate21_6_icde_ready.py`
- [x] `experiments/scripts/summarize_gate21_6_icde_ready.py`
- [x] `experiments/scripts/run_gate21_6_directed_skeleton_stability.py`
- [x] `experiments/scripts/run_gate21_6_feature_ablation_safe.py`
- [x] `experiments/scripts/run_gate21_6_feature_adapter_package.py`
- [x] `experiments/scripts/run_gate21_6_external_baselines_tp.py`
- [x] `experiments/scripts/run_gate21_6_standard_condensation_baselines.py`
- [x] `experiments/scripts/run_gate21_6_cross_dataset_auto_channel.py`

## Required Top-Level Outputs
- [x] `planned_runs.csv`
- [x] `gate21_6_directed_skeleton_by_method.csv`
- [x] `gate21_6_graph_seed_stability.csv`
- [x] `gate21_6_feature_ablation_safe.csv`
- [x] `gate21_6_feature_adapter_by_method.csv`
- [x] `gate21_6_adapter_manifest_index.csv`
- [x] `gate21_6_external_tp_by_method.csv`
- [x] `gate21_6_external_tp_artifact_audit.csv`
- [x] `gate21_6_standard_condensation_by_method.csv`
- [x] `gate21_6_storage_only_baselines.csv`
- [x] `gate21_6_system_resource_by_stage.csv`
- [x] `gate21_6_metapath_cache_audit.csv`
- [x] `gate21_6_coverage_diagnostics.csv`
- [x] `gate21_6_cross_dataset_auto_channel_by_method.csv`
- [x] `gate21_6_decision.json`
- [x] `gate21_6_decision.md`
- [x] `gate21_6_main_table_official.csv`
- [x] `gate21_6_adapter_table.csv`
- [x] `gate21_6_external_tp_table.csv`
- [x] `gate21_6_storage_system_table.csv`
- [x] `gate21_6_ablation_table.csv`

## Explicit Non-Claims
- Path-aware AP/PV pruning success is not claimed.
- FreeHGC/HGCond success is not claimed without local external dependency execution.
- Adapter package ratios are reported only with manifest completeness fields.
