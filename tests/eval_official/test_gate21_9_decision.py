from __future__ import annotations

import hashlib
import math


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_ready_flags_cannot_pass_with_nan_or_placeholder_metrics() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    decision = gate21_9_decision(
        external_tp_rows=[
            {
                "method": "Random-HG-TP",
                "graph_seed": 1,
                "training_seed": 1,
                "training_executed": True,
                "official_hgb_exported": True,
                "official_sehgnn_unmodified": True,
                "test_micro_f1": math.nan,
                "test_macro_f1": "",
            }
        ],
        cross_dataset_rows=[
            {
                "dataset": "ACM",
                "method": "full-native-SeHGNN",
                "training_executed": True,
                "success": True,
                "test_micro_f1": "",
                "test_macro_f1": "",
            }
        ],
    )

    assert decision["flags"]["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] is False
    assert decision["flags"]["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] is False


def test_5x5_ready_requires_five_graph_and_training_seeds_unless_deterministic_proof() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import external_tp_method_status

    smoke = [
        {
            "method": "Random-HG-TP",
            "graph_seed": 1,
            "training_seed": 1,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "training_executed": True,
            "test_micro_f1": 0.8,
            "test_macro_f1": 0.7,
        }
    ]
    deterministic = [
        {
            "method": "Random-HG-TP",
            "graph_seed": 1,
            "training_seed": seed,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "training_executed": True,
            "test_micro_f1": 0.8,
            "test_macro_f1": 0.7,
            "sampler_deterministic": True,
            "deterministic_export_hash_unit_test_pass": True,
        }
        for seed in range(1, 6)
    ]

    assert external_tp_method_status(smoke, "Random-HG-TP")["ready_5x5_flag"] is False
    assert external_tp_method_status(deterministic, "Random-HG-TP")["ready_5x5_flag"] is True


def test_freehgc_tp_ready_requires_official_export_and_real_training_metrics() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    decision = gate21_9_decision(
        freehgc_tp_rows=[
            {
                "method": "FreeHGC-TP",
                "official_hgb_exported": False,
                "official_sehgnn_unmodified": True,
                "training_executed": True,
                "test_micro_f1": 0.8,
                "test_macro_f1": 0.7,
            }
        ]
    )

    assert decision["flags"]["FREEHGC_TP_TASK_RESULTS_READY"] is False
    assert decision["flags"]["FREEHGC_TP_ADAPTER_IMPLEMENTED"] is False


def test_freehgc_tp_hard_gap_requires_specific_reason_not_generic_placeholder() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    generic = gate21_9_decision(freehgc_tp_rows=[{"method": "FreeHGC-TP", "failure_type": "adapter_not_implemented"}])
    concrete = gate21_9_decision(
        freehgc_tp_rows=[
            {
                "method": "FreeHGC-TP",
                "failure_type": "hard_incompatibility",
                "hard_incompatibility_reason": "freehgc_output_not_exportable_to_official_hgb",
                "failure_message": "Upstream FreeHGC output lacks support-node identity needed for official HGB export.",
            }
        ]
    )

    assert generic["flags"]["FREEHGC_TP_HARD_GAP_REPORTED"] is False
    assert concrete["flags"]["FREEHGC_TP_HARD_GAP_REPORTED"] is True


def test_apv16_graph_seed_status_distinguishes_deterministic_empirical_and_not_validated() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import apv16_graph_seed_status

    deterministic = apv16_graph_seed_status(
        {
            "method": "HeSF-RCS-APV16",
            "sampler_deterministic": True,
            "graph_seed_ignored_by_design": True,
            "deterministic_export_hash_unit_test_pass": True,
            "empirical_graph_seed_count": 1,
        }
    )
    empirical = apv16_graph_seed_status(
        {
            "method": "HeSF-RCS-APV16",
            "sampler_deterministic": False,
            "empirical_graph_seed_count": 5,
            "empirical_graph_seed_stability_pass": True,
        }
    )
    missing = apv16_graph_seed_status({"method": "HeSF-RCS-APV16", "empirical_graph_seed_count": 1})

    assert deterministic["apv16_stability_mode"] == "deterministic_proof"
    assert empirical["apv16_stability_mode"] == "empirical_5x5"
    assert missing["apv16_stability_mode"] == "not_validated"


def test_metapath_and_cache_pass_requires_non_empty_hashes_and_assertions() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    empty = gate21_9_decision(
        metapath_rows=[{"feature_tensor_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}],
        cache_assertion_rows=[{"assertion": "cache_hash_non_empty", "assertion_pass": True, "cache_file_hash": ""}],
    )
    real = gate21_9_decision(
        metapath_rows=[{"feature_tensor_hash": _sha(b"tensor"), "feature_tensor_bytes": 32, "metapath_key": "AP"}],
        cache_assertion_rows=[
            {"assertion": "cache_hash_non_empty", "assertion_pass": True, "cache_file_hash": _sha(b"cache")},
            {"assertion": "APV12_vs_APV16_cache_hash_differs", "assertion_pass": True, "cache_file_hash": _sha(b"cache2")},
        ],
    )

    assert empty["flags"]["METAPATH_INTROSPECTION_PASS"] is False
    assert empty["flags"]["CACHE_HASH_REAL_PASS"] is False
    assert real["flags"]["METAPATH_INTROSPECTION_PASS"] is True
    assert real["flags"]["CACHE_HASH_REAL_PASS"] is True


def test_feature_ablation_ready_requires_apv12_and_apv16_key_task_metrics() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    key_transforms = ["raw", "zero-paper-preserve-dim", "zero-term-preserve-dim", "zero-all-support-preserve-dim", "paper-random-projection64"]
    rows = [
        {
            "method": method,
            "feature_transform": transform,
            "training_executed": True,
            "test_micro_f1": 0.9,
            "test_macro_f1": 0.88,
        }
        for method in ("HeSF-RCS-APV12", "HeSF-RCS-APV16")
        for transform in key_transforms
    ]

    assert gate21_9_decision(feature_ablation_rows=rows)["flags"]["FEATURE_ABLATION_TASK_RESULTS_READY"] is True
    assert gate21_9_decision(feature_ablation_rows=rows[:-1])["flags"]["FEATURE_ABLATION_TASK_RESULTS_READY"] is False


def test_cross_dataset_ready_requires_acm_and_imdb_successful_minimal_rows() -> None:
    from hesf_coarsen.eval.official.gate21_9_decision import gate21_9_decision

    rows = [
        {
            "dataset": dataset,
            "method": method,
            "training_executed": True,
            "success": True,
            "test_micro_f1": 0.7,
            "test_macro_f1": 0.68,
        }
        for dataset in ("ACM", "IMDB")
        for method in ("full-native-SeHGNN", "export-full-SeHGNN", "H6-node30", "HeSF-RCS-auto structural30")
    ]

    assert gate21_9_decision(cross_dataset_rows=rows)["flags"]["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] is True
    assert gate21_9_decision(cross_dataset_rows=rows[:-1])["flags"]["CROSS_DATASET_AUTO_CHANNEL_TASK_RESULTS_READY"] is False
