from __future__ import annotations

from typing import Any


def build_dblp_channel_removal_probes() -> list[dict[str, Any]]:
    specs = [
        ("full_H6_candidate", "H6-relgrid-APPA100-PVVP100-PTTP100", 0.9399, 0.9361, 1.0000, "reference candidate keeps all directed channels"),
        ("drop_PTTP", "H6-relgrid-APPA100-PVVP100-PTTP00", 0.9418, 0.9380, 0.1988, "PT/TP removal does not hurt validation"),
        ("drop_PVVP", "H6-relgrid-APPA100-PVVP00-PTTP00", 0.9080, 0.9021, 0.1372, "PV/VP removal hurts validation reachability"),
        ("drop_PAVP", "H6-relgrid-AP100-PA00-PV100-VP00-PTTP00", 0.9387, 0.9345, 0.1195, "feedback removal is acceptable at smaller budget"),
        ("drop_AP", "H6-relgrid-AP00-PA100-PV100-VP100-PTTP00", 0.6620, 0.6110, 0.1020, "AP is a hard bottleneck"),
        ("drop_PV", "H6-relgrid-AP100-PA100-PV00-VP100-PTTP00", 0.9182, 0.9120, 0.1490, "PV is a second bottleneck"),
        ("AP100-PV100-PA00-VP00-PTTP00", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00", 0.9396, 0.9356, 0.1195, "APV12 validation skeleton"),
        ("AP100-PV100-PA50-VP50-PTTP00", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00", 0.9431, 0.9397, 0.1592, "APV16 validation skeleton"),
        ("AP100-PV100-PA00-VP00-PTTP05", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP05", 0.9404, 0.9360, 0.1484, "small PT/TP tolerance probe"),
        ("AP100-PV75-PA00-VP00-PTTP00", "H6-dirskel-AP100-PA00-PV75-VP00-PTTP00", 0.9325, 0.9280, 0.1160, "PV75 reachability degradation probe"),
    ]
    return [
        {
            "dataset": "DBLP",
            "probe_name": name,
            "canonical_method": canonical,
            "metric_split": "validation",
            "uses_test_metrics": False,
            "validation_micro_f1": micro,
            "validation_macro_f1": macro,
            "structural_storage_ratio": structural,
            "probe_interpretation": note,
        }
        for name, canonical, micro, macro, structural, note in specs
    ]
