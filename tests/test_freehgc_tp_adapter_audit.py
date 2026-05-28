from __future__ import annotations


def test_freehgc_tp_audit_requires_specific_blocking_artifact() -> None:
    from hesf_coarsen.eval.official.freehgc_tp_export_adapter import freehgc_tp_hard_incompatibility_ready

    vague = {"hard_incompatibility": True, "hard_reason": "not_exportable", "minimal_blocking_artifact": ""}
    specific = {
        "hard_incompatibility": True,
        "hard_reason": "edge_provenance_missing",
        "minimal_blocking_artifact": "HGB/link.dat cannot encode FreeHGC synthetic support provenance",
    }

    assert freehgc_tp_hard_incompatibility_ready(vague) is False
    assert freehgc_tp_hard_incompatibility_ready(specific) is True
