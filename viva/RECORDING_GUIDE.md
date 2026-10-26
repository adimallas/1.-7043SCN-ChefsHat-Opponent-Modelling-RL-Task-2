# Record your own viva

The final assessed recording must be your own 3–5 minute explanation, with a trained-agent gameplay demonstration. The supplied PowerPoint and script are preparation materials, not a substitute for the recording. No synthetic voice/video is presented as the student.

Use screen recording in PowerPoint, Teams, OBS or your university-approved tool. Show your face if required by your module guidance. Open the repository, deck and a terminal before starting. Use the completed checkpoints delivered here; do not rerun lengthy training during the recording.

Run a real demonstration from the repository root:

```bash
python scripts/demo.py --checkpoint checkpoints/dqn_aware_seed17.pt --seed 777888 --seat 0 --delay 0.15
```

The optional delay paces the display for viewers; it does not change gameplay or the selected actions. Explain the own-hand, legal-policy choices and selected-action lines. Show the final natural match completion, real finishing position and official points. The public JSONL/transcript evidence is saved under `viva/gameplay/`. Do not describe a transcript or animation as a captured live student recording.

Aim for approximately 4 minutes of explanation plus 30–45 seconds of demonstration. The speaking script includes suggested time blocks. Check that every numerical statement matches `RESULTS.md`, not an example or earlier quick run.

Before submission, export/play the recording and check audio, screen readability, timing and the game completion. Upload it to an approved platform, set permissions so assessors can view it, and add its real URL to README.md. Publish the repository and verify assessor access. This package deliberately contains no fabricated video link.

Be prepared to explain legal masking, Double DQN targets, replay, epsilon decay, opponent features, the progress restriction, confidence intervals and why the experiments do not establish universal superiority. Review the AI-assisted source yourself and follow the university’s rules for acknowledging assistance.
