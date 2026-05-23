from __future__ import annotations

from hesf_coarsen.task_first.units.base import SupportUnit


def make_union_units(*unit_sets: list[SupportUnit], deduplicate: bool = True) -> list[SupportUnit]:
    out: list[SupportUnit] = []
    seen: set[tuple[int, ...]] = set()
    for units in unit_sets:
        for unit in units:
            key = tuple(int(node) for node in unit.member_nodes)
            if bool(deduplicate) and key in seen:
                continue
            seen.add(key)
            out.append(unit)
    return out
