from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import numpy as np


def test_model_class_adapter_rejects_imdb_instead_of_dblp_fallback() -> None:
    from hesf_coarsen.eval.official.sehgnn_export_runner import resolve_supported_model_dataset

    assert resolve_supported_model_dataset("DBLP") == "DBLP"
    assert resolve_supported_model_dataset("ACM") == "ACM"
    with pytest.raises(ValueError, match="unsupported SeHGNN dataset config: IMDB"):
        resolve_supported_model_dataset("IMDB")


def test_sehgnn_bridge_marks_model_class_adapter_as_non_official(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.sehgnn_bridge import run_sehgnn_official

    result = run_sehgnn_official(
        tmp_path / "missing_export",
        tmp_path / "missing_repo",
        "DBLP",
        "type_0",
        1,
        {"method": "full"},
        tmp_path / "out",
    )

    assert result["bridge_type"] == "model_class_only"
    assert result["model_name"] == "SeHGNN-modelclass-HeSF-features"
    assert result["official_pipeline"] is False
    assert result["uses_official_preprocess"] is False
    assert result["method_label"] == "SeHGNN-modelclass-HeSF-features"
    assert result["warning"] == "WARNING: This run uses HeSF-built target feature blocks and is not the official SeHGNN HGB preprocessing pipeline."


def test_official_hgb_command_is_dataset_specific_and_has_no_imdb_fallback(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.sehgnn_native_runner import build_official_hgb_command

    imdb = build_official_hgb_command(
        dataset="IMDB",
        seed=3,
        repo_dir=Path("external/SeHGNN"),
        data_root=tmp_path,
        device="cuda",
        python_executable="python",
    )

    assert imdb.cwd == Path("external/SeHGNN") / "hgb"
    assert imdb.command[:2] == ["python", "main.py"]
    assert imdb.command[imdb.command.index("--dataset") + 1] == "IMDB"
    assert imdb.command[imdb.command.index("--n-task-layers") + 1] == "4"
    assert imdb.command[imdb.command.index("--input-drop") + 1] == "0."
    assert Path(imdb.command[imdb.command.index("--root") + 1]).is_absolute()
    assert imdb.command[imdb.command.index("--seeds") + 1] == "3"
    assert "--cpu" not in imdb.command

    relative_root = build_official_hgb_command(
        dataset="DBLP",
        seed=1,
        repo_dir=Path("external/SeHGNN"),
        data_root=Path("external/SeHGNN/data"),
        device="cuda",
        python_executable="python",
    )
    assert Path(relative_root.command[relative_root.command.index("--root") + 1]).is_absolute()

    with pytest.raises(ValueError, match="unsupported official SeHGNN HGB dataset"):
        build_official_hgb_command(
            dataset="MISSING",
            seed=1,
            repo_dir=Path("external/SeHGNN"),
            data_root=tmp_path,
            device="cuda",
            python_executable="python",
        )


def test_native_subprocess_env_adds_sparse_tools_shim_before_official_repo() -> None:
    import os

    from hesf_coarsen.eval.official.sehgnn_native_runner import native_subprocess_env

    env = native_subprocess_env(repo_dir=Path("external/SeHGNN"), base_env={"PYTHONPATH": "existing"})
    paths = env["PYTHONPATH"].split(os.pathsep)

    assert paths[0].endswith("external_patches\\sehgnn_sparse_tools_shim") or paths[0].endswith("external_patches/sehgnn_sparse_tools_shim")
    assert paths[-1] == "existing"


def test_official_stdout_parser_preserves_micro_macro_and_multilabel() -> None:
    from hesf_coarsen.eval.official.sehgnn_native_runner import parse_official_hgb_stdout

    parsed = parse_official_hgb_stdout(
        "IMDB",
        """
Restart with seed = 5
#Train 1097, #Val 274, #Test 3202
Best Epoch 17 at abcdef
\tFinal Val loss 0.6123 (63.1000, 61.2000), Test loss 0.7345 (60.3000, 58.4000)
train_acc (72.01, 70.02) val_acc (63.10, 61.20) test_acc (60.30, 58.40)
""",
    )

    assert parsed["status"] == "success"
    assert parsed["best_epoch"] == 17
    assert parsed["validation_micro_f1"] == pytest.approx(0.631)
    assert parsed["validation_macro_f1"] == pytest.approx(0.612)
    assert parsed["test_micro_f1"] == pytest.approx(0.603)
    assert parsed["test_macro_f1"] == pytest.approx(0.584)
    assert parsed["test_accuracy_if_single_label"] == ""
    assert parsed["is_multilabel"] is True
    assert parsed["loss_type"] == "bce"


def test_native_hgb_data_audit_reports_missing_official_dataset(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir

    audit = audit_native_hgb_data_dir("DBLP", tmp_path)

    assert audit["dataset"] == "DBLP"
    assert audit["data_root"] == str(tmp_path)
    assert audit["node_dat_exists"] is False
    assert audit["link_dat_exists"] is False
    assert audit["label_dat_exists"] is False
    assert audit["label_dat_test_exists"] is False
    assert audit["can_load_with_official_data_loader"] is False


def test_export_graph_to_sehgnn_hgb_writes_grouped_ids_relations_and_split_labels(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.sehgnn_native_export import export_graph_to_sehgnn_hgb
    from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec

    graph = HeteroGraph(
        num_nodes=5,
        node_type=np.array([0, 0, 1, 2, 3], dtype=np.int32),
        features={
            0: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            1: np.array([[0.5, 0.5]], dtype=np.float32),
            2: np.array([[0.2, 0.8]], dtype=np.float32),
            3: np.zeros((1, 1), dtype=np.float32),
        },
        labels=np.array([0, 1, -1, -1, -1], dtype=np.int64),
        relations={
            0: RelationAdj(np.array([0]), np.array([2]), np.array([1.0], dtype=np.float32), 0, 1, 0),
            1: RelationAdj(np.array([2]), np.array([0]), np.array([1.0], dtype=np.float32), 1, 0, 1),
            2: RelationAdj(np.array([2]), np.array([3]), np.array([1.0], dtype=np.float32), 1, 2, 2),
            3: RelationAdj(np.array([2]), np.array([4]), np.array([1.0], dtype=np.float32), 1, 3, 3),
            4: RelationAdj(np.array([3]), np.array([2]), np.array([1.0], dtype=np.float32), 2, 1, 4),
            5: RelationAdj(np.array([4]), np.array([2]), np.array([1.0], dtype=np.float32), 3, 1, 5),
        },
        relation_specs={
            0: RelationSpec(0, "AP", 0, 1),
            1: RelationSpec(1, "PA", 1, 0),
            2: RelationSpec(2, "PT", 1, 2),
            3: RelationSpec(3, "PV", 1, 3),
            4: RelationSpec(4, "TP", 2, 1),
            5: RelationSpec(5, "VP", 3, 1),
        },
    )

    manifest = export_graph_to_sehgnn_hgb(
        graph=graph,
        dataset_name="DBLP",
        target_type="A",
        output_dir=tmp_path,
        split_mode="official_trainval",
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
        test_idx=np.array([1], dtype=np.int64),
        labels=graph.labels,
        method_name="full",
        seed=1,
    )

    export_dir = Path(manifest["export_dir"])
    assert (export_dir / "node.dat").read_text(encoding="utf-8").splitlines()[:3] == [
        "0\t0\t0\t1.0,0.0",
        "1\t1\t0\t0.0,1.0",
        "2\t2\t1\t0.5,0.5",
    ]
    assert (export_dir / "link.dat").read_text(encoding="utf-8").splitlines()[0] == "0\t2\t0\t1.0"
    assert (export_dir / "label.dat").read_text(encoding="utf-8").splitlines() == ["0\t0\t0\t0"]
    assert (export_dir / "label.dat.test").read_text(encoding="utf-8").splitlines() == ["1\t1\t0\t1"]
    assert manifest["mapping_bijective"] is True
    assert manifest["relation_order_matches_official"] is True


def test_compressed_storage_ratio_fields_report_total_ratio_separately() -> None:
    from experiments.scripts.run_gate21_0_sehgnn_native_export import _compressed_method_label, _storage_ratio_fields
    from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec

    original = HeteroGraph(
        num_nodes=6,
        node_type=np.array([0, 0, 1, 1, 1, 1], dtype=np.int32),
        relations={0: RelationAdj(np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5]), np.ones(4, dtype=np.float32), 0, 1, 0)},
        relation_specs={0: RelationSpec(0, "AP", 0, 1)},
    )
    compressed = HeteroGraph(
        num_nodes=4,
        node_type=np.array([0, 0, 1, 1], dtype=np.int32),
        relations={0: RelationAdj(np.array([0, 1]), np.array([2, 3]), np.ones(2, dtype=np.float32), 0, 1, 0)},
        relation_specs={0: RelationSpec(0, "AP", 0, 1)},
    )

    fields = _storage_ratio_fields(original, compressed, target_type=0, export_file_bytes=20, native_full_file_bytes=100, method="H6-node30")

    assert fields["support_node_ratio"] == 0.5
    assert fields["support_edge_ratio"] == 0.5
    assert fields["total_storage_ratio_vs_full_graph"] == 0.6
    assert fields["export_file_bytes"] == 20
    assert fields["native_full_file_bytes"] == 100
    assert _compressed_method_label("H6") == "H6-node30"
    assert _compressed_method_label("flatten") == "flatten-node30"
    assert _compressed_method_label("typedhash") == "TypedHash-node30"
    assert _compressed_method_label("target-only") == "target-only"


def test_gate21_0_summary_fails_before_export_when_native_missing(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_0_sehgnn_native_export import summarize_gate21_0

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    with (native_dir / "native_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "seed",
                "status",
                "command",
                "best_epoch",
                "validation_micro_f1",
                "validation_macro_f1",
                "test_micro_f1",
                "test_macro_f1",
                "test_accuracy_if_single_label",
                "is_multilabel",
                "loss_type",
                "train_time_sec",
                "peak_memory_mb",
                "stdout_path",
                "stderr_path",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerow({"dataset": "DBLP", "seed": 1, "status": "failed_dependency", "error_message": "missing official data"})

    result = summarize_gate21_0(tmp_path)

    assert result["decision"] == "NATIVE_SEHGNN_REPRO_FAIL"
    assert result["native_repro_pass"] is False
    assert result["export_full_fidelity_pass"] is False
    assert result["compressed_eval_allowed"] is False
    assert result["uses_official_main_py"] is True
    assert result["uses_model_class_adapter_only"] is False
    assert result["imdb_uses_dblp_fallback"] is False
    assert json.loads((tmp_path / "gate21_0_result.json").read_text(encoding="utf-8"))["decision"] == "NATIVE_SEHGNN_REPRO_FAIL"


def test_gate21_0_summary_requires_all_native_seed_runs_success(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_0_sehgnn_native_export import summarize_gate21_0

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    fieldnames = [
        "dataset",
        "seed",
        "status",
        "command",
        "best_epoch",
        "validation_micro_f1",
        "validation_macro_f1",
        "test_micro_f1",
        "test_macro_f1",
        "test_accuracy_if_single_label",
        "is_multilabel",
        "loss_type",
        "train_time_sec",
        "peak_memory_mb",
        "stdout_path",
        "stderr_path",
        "error_message",
    ]
    with (native_dir / "native_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in ("DBLP", "ACM"):
            for seed in range(1, 6):
                writer.writerow({"dataset": dataset, "seed": seed, "status": "success", "test_micro_f1": "0.95", "test_macro_f1": "0.94"})
        for seed in range(1, 5):
            writer.writerow({"dataset": "IMDB", "seed": seed, "status": "success", "test_micro_f1": "0.70", "test_macro_f1": "0.67"})

    result = summarize_gate21_0(tmp_path)

    assert result["decision"] == "NATIVE_SEHGNN_REPRO_FAIL"
    assert result["native_repro_pass"] is False


def test_gate21_0_summary_requires_all_compressed_methods_for_ready_decision(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_0_sehgnn_native_export import summarize_gate21_0

    native_dir = tmp_path / "native"
    fidelity_dir = tmp_path / "fidelity"
    compressed_dir = tmp_path / "compressed"
    native_dir.mkdir()
    fidelity_dir.mkdir()
    compressed_dir.mkdir()
    native_fields = [
        "dataset",
        "seed",
        "status",
        "command",
        "best_epoch",
        "validation_micro_f1",
        "validation_macro_f1",
        "test_micro_f1",
        "test_macro_f1",
        "test_accuracy_if_single_label",
        "is_multilabel",
        "loss_type",
        "train_time_sec",
        "peak_memory_mb",
        "stdout_path",
        "stderr_path",
        "error_message",
    ]
    with (native_dir / "native_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=native_fields)
        writer.writeheader()
        for dataset in ("DBLP", "ACM", "IMDB"):
            for seed in range(1, 6):
                writer.writerow({"dataset": dataset, "seed": seed, "status": "success", "test_micro_f1": "0.95", "test_macro_f1": "0.94"})
    with (fidelity_dir / "gate21_0_sehgnn_full_fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "fidelity_pass"])
        writer.writeheader()
        for dataset in ("DBLP", "ACM", "IMDB"):
            for seed in range(1, 6):
                writer.writerow({"dataset": dataset, "seed": seed, "fidelity_pass": "True"})
    with (compressed_dir / "gate21_0_compressed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "method", "status"])
        writer.writeheader()
        writer.writerow({"dataset": "DBLP", "seed": 1, "method": "H6-node30", "status": "success"})
    with (compressed_dir / "gate21_0_compressed_storage_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method"])
        writer.writeheader()
        writer.writerow({"dataset": "DBLP", "method": "H6-node30"})

    result = summarize_gate21_0(tmp_path)

    assert result["native_repro_pass"] is True
    assert result["export_full_fidelity_pass"] is True
    assert result["decision"] == "EXPORT_FULL_FIDELITY_PASS_COMPRESSED_READY"
    assert result["required_compressed_methods_present"] is False
