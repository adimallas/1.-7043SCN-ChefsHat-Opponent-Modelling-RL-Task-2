#!/usr/bin/env python3
"""Build and execute the review notebook against saved genuine artefacts."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.cells = [
    nbf.v4.new_markdown_cell("# Opponent-aware Double DQN in Chef’s Hat\n\nTask 2, 7043SCN. Student: Sampath Adimalla, ID 17522555, opponent-modelling variant. This review notebook loads the saved experiment CSVs and checkpoints, then recomputes the analysis figures and tables."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nROOT = Path.cwd()\nif not (ROOT / 'configs/standard.yaml').exists():\n    ROOT = ROOT.parent\nfrom sampath_rl.core import ACTIONS, SingleAgentEnv, UPSTREAM_SOURCE\nfrom sampath_rl.agents import DQNAgent, QNetwork\nfrom sampath_rl.experiment import make_opponents\nfrom sampath_rl.policy import progress_mask\nprint('Assigned variant:', 17522555 % 7)\nprint('Official actions:', len(ACTIONS))\nprint('Engine source:', UPSTREAM_SOURCE)\nprint('Network parameters:', sum(p.numel() for p in QNetwork().parameters()))\nprint(json.loads((ROOT/'docs/hardware_software.json').read_text()))"),
    nbf.v4.new_markdown_cell("## Observation and action handling\n\nThe 64-input vector contains the acting player’s own hand histogram and public state. The aware variant enables 15 public-behaviour statistics; other DQN variants zero them. The original official legality is separate from the shared empty-board progress constraint. See `docs/PASS_LOOP_DIAGNOSIS.md`."),
    nbf.v4.new_code_cell("env = SingleAgentEnv(make_opponents('mixed', 777888), seat=0, aware=True)\nstate, official_mask = env.reset(seed=777888)\npolicy_mask = progress_mask(state, official_mask)\nprint('State shape:', state.shape, 'official mask shape:', official_mask.shape)\nprint('Official choices:', [ACTIONS[i] for i in np.flatnonzero(official_mask)])\nprint('Policy choices:', [ACTIONS[i] for i in np.flatnonzero(policy_mask)])\nprint('Public opponent features:', state[47:62])"),
    nbf.v4.new_markdown_cell("## Training protocol\n\nThree DQN variants were trained for 600 matches at each of three independent seeds. Baselines are not trained. Final checkpoints, not held-out-selected checkpoints, are evaluated. The exact configuration and measured completed budgets follow."),
    nbf.v4.new_code_cell("import yaml\nprint(yaml.safe_load((ROOT/'configs/standard.yaml').read_text()))\nmetadata = [json.loads(p.read_text()) for p in sorted((ROOT/'results/training').glob('*_metadata.json'))]\ndisplay(pd.DataFrame([{k:m.get(k) for k in ['condition','training_seed','completed_episodes','steps','duration_seconds','status']} for m in metadata]))"),
    nbf.v4.new_markdown_cell("## Held-out results\n\nThese tables are generated from match-level CSVs. Score is the official 3–0 finishing-point metric. The speed-normalised compatibility statistic is derived from older documentation, not an emitted V3 engine field. Paired uncertainty clusters by deal seed, retaining all seats and checkpoints."),
    nbf.v4.new_code_cell("overall = pd.read_csv(ROOT/'results/summaries/summary_overall.csv')\nby_opponent = pd.read_csv(ROOT/'results/summaries/summary_by_opponent.csv')\npaired = pd.read_csv(ROOT/'results/summaries/paired_differences.csv')\ndisplay(overall)\ndisplay(by_opponent)\ndisplay(paired)"),
    nbf.v4.new_code_cell("from IPython.display import Image, display\nfor name in ['learning_curves','win_rates','opponent_heatmap','ablation']:\n    display(Image(filename=str(ROOT/'results/figures'/f'{name}.png')))"),
    nbf.v4.new_markdown_cell("## Complete official match with a trained checkpoint\n\nThe following executes one natural complete game, without exploration or learning. It is a gameplay test, not a synthetic student viva recording. The fixed demonstration seed is not chosen for a favourable outcome."),
    nbf.v4.new_code_cell("agent = DQNAgent.load(ROOT/'checkpoints/dqn_aware_seed17.pt')\nenv = SingleAgentEnv(make_opponents('mixed',777888),seat=0,aware=True,shaping=0.0)\nstate, mask = env.reset(seed=777888)\ndecisions=[]\nwhile True:\n    mask=progress_mask(state,mask)\n    action=agent.act(state,mask,epsilon=0.0)\n    decisions.append(ACTIONS[action])\n    state,reward,terminal,truncated,info=env.step(action)\n    mask=info['action_mask']\n    if terminal or truncated:\n        break\nassert terminal and not truncated\nprint('Learner actions:', decisions)\nprint({k:v for k,v in info.items() if k != 'action_mask'})"),
    nbf.v4.new_markdown_cell("## Interpretation and submission steps\n\nRead `RESULTS.md` and `docs/LIMITATIONS.md` before making claims. Lower loss is not evidence of better held-out play, and three training seeds do not establish general superiority. The same-checkpoint feature-off ablation introduces distribution shift. Publish using an authorised account, verify assessor access, record your own 3–5 minute viva and add its real accessible URL. No remote repository or student recording is claimed by this notebook."),
]
path = ROOT / 'notebooks' / 'project_review.ipynb'
path.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata':{'path': str(ROOT)}}).execute()
nbf.write(nb, path)
print(f'Executed notebook saved: {path}')
