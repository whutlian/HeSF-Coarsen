# Gate13 Task-First Debug Experiment Plan

Goal: determine whether HeSF-TC can beat strong support-only baselines under the same `hettree_lite` diagnostic evaluator.

Scope:
- Audit Gate12 code paths and evaluator protocol.
- Add target-aware candidate sources.
- Add honest pair-delta modes: `local_surrogate`, `exact_pair_isolated`, `response_signature`.
- Add cross-anchor/class-context coverage penalties.
- Add zero-footprint support purity policies.
- Run full graph ceiling, candidate ablation, pair-delta ablation, coverage/purity ablation, relation-response activation, ratio-budget sanity, support-only baselines, and final Gate13 table.
- Summarize go/no-go in `gate13_decision.md`.

Non-goals:
- Do not revive Hybrid-B target-selection/meta-reconstruction/distillation.
- Do not claim official SeHGNN/HETTREE/FreeHGC performance.
- Do not modify the preservation-first HeSF-LVC-P/S mainline.
