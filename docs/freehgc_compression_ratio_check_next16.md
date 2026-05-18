# FreeHGC Compression Ratio Check

Checked on 2026-05-18 against:

- Paper HTML: https://ar5iv.org/html/2412.16250v1
- Code: https://github.com/PKU-DAIR/FreeHGC

Conclusion:

- The paper definition is whole-graph/type-wise. It states that all node types are condensed according to the condensation ratio, so the condensed graph has the same ratio type by type and therefore the same total-node ratio in the ideal setting.
- The medium-scale code starts the budget from target-type training labels: `Base` computes per-class budgets from `len(labels[train_nid]) * args.reduction_rate`.
- The code then converts that into an actual target-type keep ratio and applies it to other node types. It also prints a final whole-graph `real_reduction_rate` as `sum_nodes / sum(node_type_nodes.values())` after constructing the condensed graph.
- Therefore, for our ACM/DBLP/IMDB 1.2%, 2.4%, 4.8%, and 9.6% experiment, the defensible reporting convention is whole-graph node compression ratio, with actual measured ratio reported beside the requested target ratio.

Implication for the SeHGNN run:

- `target_ratio` is the requested whole-graph node ratio.
- `actual_ratio` is computed as `coarse.num_nodes / original.num_nodes`.
- The Next16 SeHGNN summary uses actual ratios around 0.012, 0.024, 0.048, and 0.096 for ACM, DBLP, and IMDB.
