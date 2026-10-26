# Sources and provenance

The final backend is the official Chef’s Hat Gym `core.game_env.game.Game` implementation at base commit `bf882e55eccd537d189f8b99fef1ad104339c92b`, plus the documented one-line patch changing `if jokers == 1` to `if jokers >= 1`. It enables the existing `C0;Q0;J1` action when two jokers are held; it adds no action ID. The exact source code and patch are inspected and pinned rather than assuming the older documentation API matches the current version. `docs/environment_provenance.json` records the source-rules SHA-256, patch SHA-256, and exact diff.

[1]: https://github.com/pablovin/ChefsHatGYM "Official Chef’s Hat Gym repository"
[2]: https://chefshatgym.readthedocs.io/en/latest/03_environment.html "Chef’s Hat environment documentation"

The repository describes V3 and a synchronous game engine coordinated by rooms [1]. The documentation describes an older observation/info API, 200 actions, 3–0 finishing points, and a performance-score formula [2]. Implementation-specific differences are documented in `ENVIRONMENT_COMPATIBILITY.md`.

Upstream packaging declares an MIT licence, while its README also states research-only/non-commercial and no-redistribution conditions for examples. This package does not redistribute upstream game source, assets, or the supplied university assignment brief. The installation script retrieves the pinned official source externally and explicitly applies the documented joker patch.

The project was generated with Manus assistance. Experimental outputs are produced by executing the patched official backend; no performance figures are invented. The 600-match, three-seed final training budget is still in progress, so no completed final performance set is claimed. Any final findings are conditional on both this patched backend and the separate shared progress guard. The student must review, understand, and disclose assistance in accordance with applicable university policy.

Double DQN follows van Hasselt, Guez and Silver’s online-selection/target-evaluation construction, proposed to reduce Q-value overestimation [3]. This project does not claim to reproduce their Atari results.

[3]: https://ojs.aaai.org/index.php/AAAI/article/view/10295 "Deep Reinforcement Learning with Double Q-Learning, AAAI 2016"
