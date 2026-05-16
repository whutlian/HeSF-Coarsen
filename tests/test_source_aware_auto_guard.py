import numpy as np

from hesf_coarsen.scoring.guards import apply_source_aware_auto_guard


def test_source_aware_auto_guard_triggers_on_onehop_pollution():
    scored = np.array(
        [[0, 1, 0.1], [0, 2, 0.2], [1, 2, 0.3], [2, 3, 0.4]],
        dtype=np.float64,
    )
    terms = {"spec": np.array([0.1, 0.2, 2.0, 2.2], dtype=np.float32)}
    sources = {(0, 1): "bucket", (0, 2): "bucket", (1, 2): "onehop", (2, 3): "onehop"}

    filtered, _terms, diagnostics = apply_source_aware_auto_guard(
        scored,
        terms,
        source_lookup=lambda left, right: sources[(left, right)],
        config={
            "source_aware_guard": {
                "enabled": True,
                "mode": "auto",
                "trigger": {
                    "onehop_selected_share_above": 0.30,
                    "onehop_avg_delta_spec_ratio_to_bucket_above": 4.0,
                    "onehop_delta_spec_tail_quantile": 0.95,
                },
                "action": {
                    "onehop_topk_per_node": 2,
                    "reject_if_delta_spec_above_bucket_q95": True,
                    "fallback_max_selected_share": 0.05,
                },
            }
        },
    )

    assert diagnostics["guard_triggered"] is True
    assert "onehop" in diagnostics["trigger_reason"]
    assert diagnostics["rejected_by_spec_count"] == 2
    assert filtered.shape[0] == 2
    assert diagnostics["source_selected_share_before"]["onehop"] == 0.5
    assert diagnostics["source_selected_share_after"].get("onehop", 0.0) == 0.0


def test_source_aware_auto_guard_stays_inactive_when_sources_are_comparable():
    scored = np.array([[0, 1, 0.1], [1, 2, 0.2]], dtype=np.float64)
    terms = {"spec": np.array([0.2, 0.22], dtype=np.float32)}
    sources = {(0, 1): "bucket", (1, 2): "onehop"}

    filtered, _terms, diagnostics = apply_source_aware_auto_guard(
        scored,
        terms,
        source_lookup=lambda left, right: sources[(left, right)],
        config={
            "source_aware_guard": {
                "enabled": True,
                "mode": "auto",
                "trigger": {
                    "onehop_selected_share_above": 0.30,
                    "onehop_avg_delta_spec_ratio_to_bucket_above": 4.0,
                    "onehop_delta_spec_tail_quantile": 0.95,
                },
            }
        },
    )

    assert diagnostics["guard_triggered"] is False
    assert diagnostics["rejected_by_spec_count"] == 0
    assert filtered.shape[0] == 2
