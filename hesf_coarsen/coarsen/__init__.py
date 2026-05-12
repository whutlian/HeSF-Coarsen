from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph, coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment

__all__ = [
    "Assignment",
    "LevelResult",
    "coarsen_graph",
    "coarsen_graph_chunked",
    "run_multilevel_coarsening",
]


def __getattr__(name: str):
    if name in {"LevelResult", "run_multilevel_coarsening"}:
        from hesf_coarsen.coarsen.multilevel import LevelResult, run_multilevel_coarsening

        values = {
            "LevelResult": LevelResult,
            "run_multilevel_coarsening": run_multilevel_coarsening,
        }
        return values[name]
    raise AttributeError(name)
