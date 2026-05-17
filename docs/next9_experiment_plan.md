# Next9 Experiment Plan

P0 creates the HGB paper-final tables from the Next8 final gap summary, preserving per-dataset and per-seed views and making full tuned RGCN stronger on task F1 explicit.

P1 adds relation-sensitive rebuttal diagnostics for flatten-sum and H6-no-spec: relation energy error, relation mass drift, coarse edge collapse, bounded metapath sanity, and checkpoint/refine masking.

P2 adds optional guard code paths. Spectral guard rejects or downranks excessive fused-operator distortion. Source-aware guard triggers only when onehop candidates show spectral pollution relative to bucket candidates.

P3 instruments OGBN aggregation with fine-grained timers and per-relation throughput fields. OGBN-MAG remains system/protocol evidence, not task-quality evidence.

P4 builds quality-cost Pareto outputs with task, spectral/operator, runtime, memory, and compression fields.

P5 updates configs and claim-boundary docs so the repository matches the paper claim.

Final paper experiment structure:

```text
E1. HGB main table:
    P, S, H0, random, GraphZoom-style, ConvMatch-style, flatten-sum, H6, full graph references

E2. Flatten/H6 rebuttal:
    relation drift, energy error, collapse, masking/refine curve

E3. Ablation:
    no-spec, spectral-only, lambda_rel=0, optional conv/rel

E4. Quality-cost:
    task vs wall-clock/RSS/graph size

E5. OGBN system:
    scale + aggregation bottleneck breakdown

Appendix:
    guard ablation
    bounded metapath
    aggressive 0.25
```

Next10 completion pass:

- P0 extracts paper-ready rebuttal tables from `outputs/exp_next9_hgb_flatten_h6_rebuttal_20260517_summary`.
- P2 reruns the minimal HGB quality-cost set with resource logging.
- P3 reruns the complete guard ablation only as appendix validation.
- P4 reruns OGBN aggregation with fresh instrumentation.
- P5 keeps the claim boundary aligned with the generated evidence.

Local command pattern:

```powershell
$PY = 'C:\Users\slian\anaconda3\envs\pytorch\python.exe'
$DATE_TAG = Get-Date -Format yyyyMMdd
& $PY -m pytest
```

If CUDA runs OOM or exhaust VRAM, use the same module commands on the server with the same output directory names and `--device cuda`.
