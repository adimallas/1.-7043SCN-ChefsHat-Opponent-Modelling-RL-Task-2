# Reproducing the project

**Published checkpoint scope:** The public `checkpoints/*.pt` files preserve learned weights, target-network parameters, optimiser state and RNG state but intentionally omit the high-volume experience replay arrays. They support deterministic inference and evaluation, not bit-exact continuation of interrupted training. To resume with replay, use the separately retained, trusted full checkpoints from the original training run; otherwise retrain with the fixed configuration. This distinction does not alter any reported evaluation result.


Use Python 3.10 or later, Git and a CPU machine with sufficient RAM. No GPU or paid model API is required. The software/hardware observed for the current run is recorded in `hardware_software.json`; the portable lower-bound dependencies are in `pyproject.toml`, while `requirements.txt` records the observed versions. CPU PyTorch wheels may require the official CPU wheel index on your architecture.

From the extracted repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest websockets==15.0.1
python scripts/install_environment.py
python -m pytest -q
```

The install script retrieves the official environment commit `bf882e55eccd537d189f8b99fef1ad104339c92b` into `.deps/ChefsHatGYM` and explicitly applies `docs/upstream_joker_fix.patch`: `if jokers == 1` becomes `if jokers >= 1`, enabling the existing `C0;Q0;J1` action when two jokers are held. Upstream source and assets are not bundled. An existing exact checkout can instead be selected through `CHEFSHAT_SRC=/path/to/ChefsHatGYM/src`, to which the same patch is applied. `docs/environment_provenance.json` records the source-rules SHA-256 `9ac6524c4e196baf6d3549e31bd73ed3f0d19bb22a63ad9538cca438fc809ff5` and patch SHA-256 `5b038a7114080abd7efbf2816cdce322dad8d1bddbaa20945f49f3e3ae1771bb`. Upstream’s ordinary Flit installation fails in this revision; importing the explicitly patched external source is intentional.

## Inspect saved results without retraining

```bash
python -m sampath_rl.analysis
python scripts/verify_results.py
python scripts/demo.py --checkpoint checkpoints/dqn_aware_seed17.pt --seed 777888 --seat 0
```

These commands regenerate available summaries/figures and verify saved CSVs, checkpoint loading and inference. Only load trusted checkpoints; the resumable files contain Python-pickle optimiser, replay and RNG state. Saved outputs are real experiments, not placeholders; pre-patch outputs are archival rather than final evidence.

## Quick validation

```bash
python scripts/run_all.py --config configs/quick.yaml --workers 3
```

The delivered package may already contain quick outputs. Move those outputs to an archive before re-running, because scripts intentionally refuse to overwrite runs. The original pass-loop failure is preserved in `results/quick_failed/`; successful pre-patch quick validation is in `results/quick_pre_joker_patch/`; and current patched quick validation is in `results/quick/`. None is final performance data.

## Run or reproduce the full configuration

The standard configuration completed 600 matches per condition for each of seeds 17, 29 and 43: nine final runs and 5,400 training matches. All partial pre-patch training is archived in `results/pre_joker_patch/` and `checkpoints/pre_joker_patch/`. Every final condition restarted from fresh weights using its original seed after the patch. Preserve delivered outputs before starting another fresh run. The scripts refuse to overwrite saved experiments, preventing accidental destruction of evidence. Move existing `results/training`, `results/evaluation`, and `checkpoints/*.pt` aside, then run:

```bash
python scripts/run_all.py --config configs/standard.yaml --workers 3
python -m sampath_rl.analysis
python scripts/verify_results.py
```

The configuration uses three worker processes for independent experiments. Each network uses one PyTorch CPU thread. Every condition has its own RNG and CSV/checkpoint path; different experiment processes do not share Python’s random state. Any final results are conditioned on the patched backend and the separate shared progress-choice guard, not untouched-upstream compliance.

## Resume an interrupted training run

```bash
python scripts/train.py --config configs/standard.yaml --condition dqn_aware --seed 17 --resume
```

The exact same configuration is required. Replay, optimiser, online and target networks, counters and RNG states are restored. Results beyond the last recorded checkpoint boundary are trimmed. Checkpoints are written atomically. Episode deal seeds are derived directly from training seed and episode number. This resumes only the authentic patched-backend run; it does not convert archived pre-patch work into a final run. Record any interruption or configuration change rather than describing a fresh run as continuous historical work.

## Determinism

Matched network seeds and common evaluation deal seeds permit controlled comparisons. Exact floating-point identity across different CPU architectures or PyTorch releases is not guaranteed. Runtime varies with machine load. Match-level CSVs and bootstrap RNG settings provide an audit trail for summaries on the recorded software stack. The game’s private-card state is never provided to opponent-aware observation features.
