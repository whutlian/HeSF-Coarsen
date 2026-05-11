from io import StringIO

from hesf_coarsen.cli.main import build_parser
from hesf_coarsen.progress import progress_iter, progress_message


def test_progress_helpers_are_quiet_by_default():
    stream = StringIO()

    values = list(progress_iter(range(3), total=3, desc="quiet", config={}, stream=stream))
    progress_message({}, "hidden", stream=stream)

    assert values == [0, 1, 2]
    assert stream.getvalue() == ""


def test_plain_progress_backend_writes_counted_progress():
    stream = StringIO()
    config = {
        "progress": {
            "enabled": True,
            "backend": "plain",
            "min_interval_seconds": 0,
        }
    }

    values = list(progress_iter(range(3), total=3, desc="demo", config=config, stream=stream))
    progress_message(config, "stage done", stream=stream)
    output = stream.getvalue()

    assert values == [0, 1, 2]
    assert "demo" in output
    assert "3/3" in output
    assert "stage done" in output


def test_coarsen_cli_accepts_progress_override():
    parser = build_parser()
    args = parser.parse_args(
        [
            "coarsen",
            "--config",
            "configs/default.yaml",
            "--input",
            "data/tiny",
            "--output",
            "outputs/tiny",
            "--progress",
        ]
    )

    assert args.progress is True
