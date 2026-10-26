# Official environment compatibility

The final backend is the official Chef’s Hat `Game` implementation at base revision `bf882e55eccd537d189f8b99fef1ad104339c92b`, plus the one-line patch in `upstream_joker_fix.patch`. This is explicitly a patched official environment, not the untouched upstream version. `environment_provenance.json` records the base commit, exact patch hash and diff. Full details and preserved failures are in `TWO_JOKER_FIX.md`.

Upstream setup metadata describes version 3.0.0.1. Its Flit build expects a missing `chefshatgym` module and fails with `ValueError: No file/folder found for module chefshatgym`; the failure log is saved. The installation script retrieves the pinned source and applies the patch without redistributing game source/assets. Optional TensorFlow agents are not used.

Before training, the official local-room test passed in 2.58 seconds. The complete upstream test directory subsequently reported two passed and one skipped. The project adapter’s integration tests validate complete four-player and two-match games, role assignment/card exchange, masks and illegal-action handling. Final patched-backend validation is recorded separately in `project_tests_final.log` and `upstream_patched_suite.log`.

The current engine uses `Game`, `Match`, and `Round`, string actions such as `C11;Q2;J0` and `pass`, and an authoritative 200-action catalogue. The patch only changes joker-only availability on an empty board from exactly one joker to at least one; it retains the existing `C0;Q0;J1` action ID. It does not reimplement the game, alter scoring, invent outcomes, or add actions.

The older online documentation describes a different observation/info interface. The current engine provides an acting player’s padded hand, board and possible actions. Actual finishing points are 3, 2, 1 and 0. It does not emit the older `Game_Performance_Score`; `performance_score` in this package is explicitly derived as `10 * match_points / rounds` for the single-match experiments, not claimed to be an emitted field.

Every policy also shares a separate empty-board progress choice constraint, documented in `PASS_LOOP_DIAGNOSIS.md`. The original engine legality is available separately; no action is substituted after execution. Final results are conditional on the common choice constraint and documented joker fix.

Role assignment and exchange are called through upstream methods on later-match transitions. Main experiments are independent single-match games; a two-match test separately validates role/exchange progression. Joker special declarations are not enabled in the adapter, limiting full multi-match equivalence with Room. Private opponent card identities are not encoded in learner state or public traces.
