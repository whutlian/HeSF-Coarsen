from __future__ import annotations

from pathlib import Path


def test_export_file_list_hash_changes_when_link_dat_changes(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.cache_hygiene import compute_export_file_list_hash

    export_dir = tmp_path / "DBLP"
    export_dir.mkdir()
    (export_dir / "node.dat").write_text("0\ta\t0\t1,0\n", encoding="utf-8")
    (export_dir / "link.dat").write_text("0\t1\t0\t1.0\n", encoding="utf-8")

    before = compute_export_file_list_hash(export_dir)
    (export_dir / "link.dat").write_text("0\t1\t0\t1.0\n1\t2\t0\t1.0\n", encoding="utf-8")

    assert compute_export_file_list_hash(export_dir) != before


def test_force_reprocess_deletes_preexisting_cache_dir(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.cache_hygiene import CacheNamespace, prepare_unique_cache_dir

    namespace = CacheNamespace("DBLP", "H6-APV-skeleton", 1, 2, "abcdef123456", tmp_path)
    cache_dir = namespace.cache_dir
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.pt").write_text("stale", encoding="utf-8")

    audit = prepare_unique_cache_dir(namespace, force_reprocess=True)

    assert audit["cache_dir_exists_before_run"] is True
    assert audit["cache_dir_deleted_before_run"] is True
    assert cache_dir.exists()
    assert not (cache_dir / "stale.pt").exists()
    assert "H6-APV-skeleton" in str(cache_dir)
    assert "graph_seed_1" in str(cache_dir)
    assert "training_seed_2" in str(cache_dir)


def test_cache_audit_marks_forced_unique_namespace_clean(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.cache_hygiene import CacheNamespace, collect_cache_audit_before_after, prepare_unique_cache_dir

    namespace = CacheNamespace("DBLP", "H6-APV-skeleton", 1, 1, "feedface", tmp_path)
    before = prepare_unique_cache_dir(namespace, force_reprocess=True)
    (namespace.cache_dir / "generated.pt").write_text("cache", encoding="utf-8")
    after = collect_cache_audit_before_after(namespace.cache_dir, before)

    assert after["cache_hygiene_pass"] is True
    assert after["cache_reused_flag"] is False
    assert after["cache_files_count_after"] == 1
