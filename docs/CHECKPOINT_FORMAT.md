# Checkpoint format and provenance

The nine published `.pt` checkpoints retain the original trained `online_state_dict`, `target_state_dict`, optimiser state, step counters, seed, configuration, RNG state and training metadata. To keep the public repository practical, their large experience-replay arrays were excluded without changing the neural-network tensors or outcome CSVs.

The original training checkpoints were created with `includes_replay=true`; public copies have `includes_replay=false` and no `replay` key. The `DQNAgent.load(path)` API supports evaluation from these compact files using its default `load_replay=False`. Loading with `load_replay=True` requires the original trusted full checkpoint and intentionally raises a helpful error when replay is absent.

**Security:** PyTorch checkpoints use Python pickle. Load only trusted files and never execute unknown checkpoint data. Recorded game results and analysis scripts are separate from the checkpoint format.

The conversion preserved each model's online tensor values exactly; it is packaging, not new training or a re-evaluation.
