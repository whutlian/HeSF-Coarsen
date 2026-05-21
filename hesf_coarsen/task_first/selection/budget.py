from __future__ import annotations


def desired_support_count(num_support: int, support_ratio: float) -> int:
    total = max(0, int(num_support))
    ratio = min(max(float(support_ratio), 0.0), 1.0)
    if total <= 0 or ratio <= 0.0:
        return 0
    return max(1, min(total, int(__import__("math").ceil(total * ratio - 1.0e-12))))


def exact_budget_from_baseline_or_ratio(
    *,
    num_support: int,
    support_ratio: float | None = None,
    baseline_support_count: int | None = None,
) -> int:
    if baseline_support_count is not None:
        return max(0, min(int(num_support), int(baseline_support_count)))
    if support_ratio is None:
        raise ValueError("either support_ratio or baseline_support_count is required")
    return desired_support_count(int(num_support), float(support_ratio))


def assert_budget_close(realized: int, desired: int, tolerance: int = 0) -> None:
    error = abs(int(realized) - int(desired))
    if error > int(tolerance):
        raise AssertionError(f"support budget mismatch: realized={realized}, desired={desired}, tolerance={tolerance}")


def budget_diagnostics(
    *,
    num_support: int,
    support_ratio: float,
    realized_support_count: int,
    tolerance: int = 0,
) -> dict[str, int | float | bool]:
    desired = desired_support_count(int(num_support), float(support_ratio))
    realized = int(realized_support_count)
    error = int(realized - desired)
    return {
        "support_node_count": int(num_support),
        "requested_support_ratio": float(support_ratio),
        "requested_support_count": int(desired),
        "realized_support_count": int(realized),
        "realized_support_ratio": float(realized / max(int(num_support), 1)),
        "support_budget_error": int(error),
        "support_budget_abs_error": int(abs(error)),
        "support_budget_exact_match": bool(abs(error) <= int(tolerance)),
    }
