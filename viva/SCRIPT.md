# Viva speaking script — approximately 4–5 minutes

This is a preparation script, not a recorded or submitted viva. Rehearse it in your own words and verify understanding. At a steady pace, allow roughly four minutes for explanation and 25–35 seconds for the actual demonstration. The matching notes are embedded in the PowerPoint.

## 1. Project question

Hello, I’m Sampath Adimalla. My student ID gives remainder one when divided by seven, so my assigned focus is opponent modelling. The project asks whether training against diverse strategies and observing public opponent behaviour improves Chef’s Hat performance. Three Double DQN conditions completed 5,400 training matches in total. Their frozen checkpoints and two baselines were then evaluated in 8,960 held-out matches, including a feature-off ablation. These are bounded CPU experiments, not a claim of state-of-the-art performance.

## 2. Environment and information

Chef’s Hat is a four-player, turn-based card game. The fixed catalogue has 200 actions, but only a subset is legal at each turn. The learner sees its own hand and public game information, never opponents’ private card identities. The official source exposed two progress problems. One required a documented one-line joker discard fix. The other is handled by the same empty-board choice constraint for every policy. Failed runs are preserved separately. The conclusions are conditional on this exact backend.

## 3. Algorithm and features

All three learned agents use the same 64-input network with two 128-unit hidden layers and 200 Q-value outputs. Double DQN selects the next legal action with the online network and evaluates it with the target network. Replay, Huber loss and target updates support learning. The aware condition activates fifteen public behaviour features, while the other conditions zero them. Random training uses random opponents; mixed and aware training share diverse opponents. Rewards combine delayed finishing points with small discard shaping. Evaluation uses unshaped outcomes.

## 4. Experiment design

Each condition trained for 600 matches at seeds seventeen, twenty-nine and forty-three. Only final-budget checkpoints were evaluated, with no selection using test results. Every configuration faced five opponent types using the same thirty-two held-out deal seeds and all four seats. The full set contains 8,960 complete evaluation matches, with zero invalid actions. Confidence intervals resample complete deal-seed blocks, retaining shared seats, checkpoints and opponent conditions. Between-training-seed variation is reported separately, because three seeds cannot establish universal superiority.

## 5. Main findings

The heuristic baseline achieved the highest observed overall win rate, about fifty-nine percent. DQN-Mixed achieved about thirty-seven percent, DQN-Aware thirty-five percent, and DQN-Random thirty-four percent. The constrained random baseline was about eleven percent. Therefore, adding explicit opponent features did not produce an overall improvement over mixed training. The learned policies did improve substantially over the random baseline, but they did not beat the stronger rule-based strategy. The repository includes the exact match counts, uncertainty intervals and source CSVs for these comparisons.

## 6. Ablation and non-stationarity

When the aware checkpoints received their behaviour features, their overall win rate was thirty-five percent. Zeroing those features reduced it to twenty-four percent. This shows that the checkpoints rely on these inputs, but zeroing them also creates distribution shift. It does not prove superiority over an independently trained mixed agent. In the corrected switching test, opponents genuinely alternate after twenty observed actions. Mixed and aware agents both won about seventeen percent, with random-trained DQN very similar. There is no clear awareness advantage in this non-stationary test.

## 7. Demonstration and conclusion

I’ll now run the saved aware checkpoint in the official game using the fixed demonstration seed seven-seven-seven-eight-eight-eight and seat zero. The saved execution completed naturally, with first place, three official points and fifteen learner decisions. One favourable demonstration does not represent the aggregate findings. Important limits are the short training budget, three seeds, hand-designed opponents and disclosed engine modifications. Longer training and recurrent opponent modelling would be useful next steps. The full code, raw results, checkpoints and genuine gameplay evidence make the experiment reproducible.

Pause and run the following real demonstration while screen recording:

```bash
python scripts/demo.py --checkpoint checkpoints/dqn_aware_seed17.pt --seed 777888 --seat 0 --delay 0.15
```

Explain one own-hand/action line and show the natural completion message and official score. Do not replace the demonstration with a fabricated result. At the end, follow your university’s required acknowledgement of AI assistance. Upload your actual recording and verify assessor permissions; no recording URL is supplied or invented here.
