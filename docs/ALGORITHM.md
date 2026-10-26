# Algorithm and design choices

## Why Double DQN?

The official environment exposes a fixed catalogue of 200 action IDs but only a state-dependent subset is legal. A compact Q-network is appropriate for this discrete space. Double DQN reduces the action-selection/evaluation coupling of ordinary DQN by choosing the next legal action using the online network and evaluating that selected action using the target network.

For a learner decision transition, the target is `r + gamma * Q_target(s_next, argmax_legal Q_online(s_next))` when non-terminal, and exactly `r` at terminal. Illegal actions are excluded from both acting and the bootstrap argmax. An all-false next mask is allowed only at terminal. Replay sampling breaks some sequential dependence, and hard target-network updates stabilise targets. These devices do not remove non-stationarity or partial observability.

Each network is a 64-input, two-hidden-layer multilayer perceptron with 128 ReLU units per hidden layer and 200 Q-value outputs. The three conditions have identical architectures. Non-aware conditions zero the same fifteen behavioural inputs rather than shrinking their network, preserving the parameter-count comparison. These zeroed-input weights still count as model parameters, although their effective contribution differs.

## Decision-time transitions

A transition spans one learner decision and all intervening opponent actions until the next learner decision. Discounting therefore operates per learner decision, not per environment turn. A final learner card discard does not terminate evaluation early: the engine completes the other players’ turns to establish the actual finishing order. The last learner transition receives the official finishing points divided by three.

The fixed training reward includes `shaping * discarded_cards`, with shaping set to 0.01 in the standard preset. It is an explicit encouragement to empty the hand, not a potential-based invariant shaping guarantee. It can affect the optimal policy, and no improvement caused specifically by this shaping is claimed without a separate reward ablation. Evaluation reports unshaped finishing outcomes and uses shaping zero.

## Opponent modelling

The aware state adds five features per opponent in cyclic seat order: passing frequency, average number discarded, mean discarded card rank, whether the last move was a pass, and last discard size. All are computed from public actions already taken. They are behavioural summaries, not access to private cards or a learned prediction of the opponent’s policy. Mixed training exposes the same diverse opponents to the aware and mixed agents.

Counts of remaining cards, finished/passed flags and public role information are observable. The acting player’s own hand is represented as rank counts; the state encoder never includes other players’ card identities. Behaviour is reset at each game.

## Limits

All nine final training runs completed 600 matches each: 5,400 training matches across three conditions and three seeds. Findings remain conditional on the official `Game` at `bf882e55eccd537d189f8b99fef1ad104339c92b` plus its documented one-line two-joker patch, and on the separate shared policy-level progress guard. Reward is delayed, behaviour is partially observed, strategies can be predictable, and the pinned implementation differs from older documentation. A decrease in training loss alone is not evidence of better play. Claims rely on frozen final-budget held-out outcomes and uncertainty estimates, not convergence or state-of-the-art performance.
