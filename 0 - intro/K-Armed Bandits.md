# Problem Definition

- A k-armed bandit is RL agent acting in a one-state MDP where k actions are possible
- Each action always returns to the same state but yields a different reward
- Example
  - Each action can be seen as a lever that lets you lose or win money
  - The goal is to maximise the money over a certain horizon
- We can define 4 different situations based on if the system is
  - Determinstic: each action does what expected
  - Known: the agent knows what is the reward when executing an action

## 4 Different Situations

- Deterministic and known
  - We take the action with highest reward

- Deterministic and unknown
  - Each action has to be executed just once to get the optimal policy (just one action)

- Non-deterministic and known
  - We take the action with highest **expected** reward
  - If the rewards are gaussians, we take the action with highest mean

- Non-deterministic and unknown
  - Ok