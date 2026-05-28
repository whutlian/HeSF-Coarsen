# Gate21.0 Requirement Checklist

- [ ] preflight repo manifest written.
- [x] native metrics CSV written.
- [x] native data audit written.
- [x] native official command manifest written.
- [x] official `hgb/main.py` command path used.
- [x] model-class adapter not used as official result.
- [x] IMDB DBLP fallback disabled.
- [x] no test leakage claimed by this native stage.
- [x] native official reproduction passed before export stage.
- [x] stopped before export/compressed if native reproduction did not pass.
- [x] export-full fidelity CSV written.
- [x] export-full fidelity passed before compressed stage.
- [x] compressed evaluation allowed only after native/export-full pass.
- [ ] required compressed methods present: H6-node30, flatten-node30, TypedHash-node30, target-only.
- [x] compressed metrics and storage audit written when compressed stage runs.

Decision: `EXPORT_FULL_FIDELITY_PASS_COMPRESSED_READY`
