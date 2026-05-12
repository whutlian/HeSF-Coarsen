from pathlib import Path

from hesf_coarsen.config import load_config


def test_ogbn_mag_cpu_chunked_config_exists_and_loads():
    path = Path("configs/ogbn_mag_A_cpu_chunked.yaml")

    config = load_config(path)

    assert path.exists()
    assert config["acceleration"]["dense_backend"] == "numpy"
    assert config["candidates"]["store_backend"] == "array"
    assert config["candidates"]["use_chunked_generation"] is True
    assert config["candidates"]["enable_partition_ann"] is False
    assert config["coarsening"]["matching_method"] == "mutual_best"
    assert config["progress"]["enabled"] is True


def test_ogbn_mag_cpu_ann_config_exists_and_loads():
    path = Path("configs/ogbn_mag_B_cpu_ann.yaml")

    config = load_config(path)

    assert path.exists()
    assert config["acceleration"]["dense_backend"] == "numpy"
    assert config["candidates"]["store_backend"] == "array"
    assert config["candidates"]["use_chunked_generation"] is True
    assert config["candidates"]["enable_partition_ann"] is True
    assert config["coarsening"]["matching_method"] == "mutual_best"
    assert config["progress"]["enabled"] is True


def test_ogbn_mag_torch_ann_config_exists_and_loads():
    path = Path("configs/ogbn_mag_C_torch_ann.yaml")

    config = load_config(path)

    assert path.exists()
    assert config["acceleration"]["dense_backend"] == "torch"
    assert config["acceleration"]["device"] == "cuda"
    assert config["candidates"]["store_backend"] == "array"
    assert config["candidates"]["use_chunked_generation"] is True
    assert config["candidates"]["enable_partition_ann"] is True
    assert config["coarsening"]["matching_method"] == "mutual_best"
    assert config["progress"]["enabled"] is True
