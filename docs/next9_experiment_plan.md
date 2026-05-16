# Next9 Experiment Plan

P0 creates the HGB paper-final tables from the Next8 final gap summary, preserving per-dataset and per-seed views and making full tuned RGCN stronger on task F1 explicit.

P1 adds relation-sensitive rebuttal diagnostics for flatten-sum and H6-no-spec: relation energy error, relation mass drift, coarse edge collapse, bounded metapath sanity, and checkpoint/refine masking.

P2 adds optional guard code paths. Spectral guard rejects or downranks excessive fused-operator distortion. Source-aware guard triggers only when onehop candidates show spectral pollution relative to bucket candidates.

P3 instruments OGBN aggregation with fine-grained timers and per-relation throughput fields. OGBN-MAG remains system/protocol evidence, not task-quality evidence.

P4 builds quality-cost Pareto outputs with task, spectral/operator, runtime, memory, and compression fields.

P5 updates configs and claim-boundary docs so the repository matches the paper claim.

Local command pattern:

```powershell
$PY = 'C:\Users\slian\anaconda3\envs\pytorch\python.exe'
$DATE_TAG = Get-Date -Format yyyyMMdd
& $PY -m pytest
```

If CUDA runs OOM or exhaust VRAM, use the same module commands on the server with the same output directory names and `--device cuda`.
