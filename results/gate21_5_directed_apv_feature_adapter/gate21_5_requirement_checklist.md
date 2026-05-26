# Gate21.5 Requirement Checklist

- [x] New Gate21.5 runners exist and support --dry-run
- [x] Directed relation specs parse and round-trip canonical method names
- [x] Directed APV skeleton methods export schema-compatible official HGB files
- [x] Relation edge retention CSV contains no missing relation ids/names/budgets/counts
- [x] Loaded relation audit confirms exported counts match loaded counts
- [x] Deterministic skeletons are not falsely marked as graph-seed unstable
- [x] Decision flags separate official structural/raw/adapter bytes
- [x] Feature loader audit proves zero/PCA/int8/fp16 transforms are loaded
- [x] Feature adapter rows are excluded from main decision and included in adapter table
- [x] Cache hygiene includes force_reprocess, unique namespace, and cache sanity
- [x] Summarizer produces by_method, raw_rows, frontiers, decision, checklist
- [x] No test labels are used for scoring, feature fitting, relation allocation, or compression decisions
