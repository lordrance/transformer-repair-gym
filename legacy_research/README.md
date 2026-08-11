# legacy_research/

Phase 0/0.5/1 code that ran on the **legacy verifiers v0 API** (`import verifiers as vf`,
`vf.Environment`, `SingleTurnEnv`, `Rubric`). Prime Intellect has deprecated v0, so none
of this is on the public execution path any more.

It is kept because the frozen historical results in `artifacts/raw/` were produced by it,
and deleting it would leave 90 trajectories with no runnable provenance. G0 indexes those
artifacts; this directory is what generated them.

`tests_v1/test_v1_migration.py::test_public_path_is_free_of_legacy_verifiers_api`
asserts that nothing under `environments/` imports the v0 API. That test is the reason
this directory exists rather than the file simply being edited in place.

- `transformer_repair_v0.py` — the Phase 0.5/1 single-turn `SingleTurnEnv` environment,
  moved out of `environments/transformer_repair/` when G1 migrated the public path to
  `verifiers.v1`.
