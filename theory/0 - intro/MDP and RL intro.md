# Reinforcement Learning: Definition

- Concerns the study of ways to teach a dynamical system to take an action $a$ while living in state $s$ following a certain goal
- Different from supervised learning because we don't have any labels
- Different from unsupervised learning because we are interested in the time evolution of the system in order to maximise a reward, not just to discover a structure itself

## Dynamic System

- The elements are:
  - State $x$: snapshot at a time instant
  - Transition function $f$: from $x_k$ to $x_{k+1}$
  - Observation function $h$: from $x_k$ to $z_k$
- Assumptions
  - Markov (**MDP**): the current state is a substitute for the entire history of the system
  - Full observability: the observation model simply returns the state
- Different problem paradigms
  - Reasoning: getting the state evolution by knowing the transition function
  - Learning: getting the transition function by knowing the state evolution
- **Reinforcement Learning Problem applied to Dynamic System**
  - Learning scenario
  - Only using past experience, compute an action at each state
  - The goal is to reach a final state (or maximise a reward)
- To summarize, the goal is to learn the policy function $\pi: S \rightarrow A$

## Formalizing MDP

- Formed by an agent and an environment, interacting with each other, generating a trajectory
  - $S_0, A_0, R_1, S_1, A_1, R_2, ..., S_n, A_n, R_{T+1}$
  - in state t, by applying action t, i get rewardt t+1
- We assume the MDP is finite and fully-observable
- $S$ finite set of states (coordinates of the agent in a grid)
- $A$ finite set of actions (directions in which the agent can go)
- Defined as a tuple $<S,A,f,r>$
  - **Deterministic** MDP
    - $ f: S \times A \rightarrow S $: transition function returns $S_{t+1} = f(S_t,A_t)$
    - $r: s \times A \rightarrow \R$: reward function returns $r(S_t,A_t)$
  - **Non-Deterministic** MDP
    - $f: S \times A \rightarrow 2^{S} $: transition function return a set of possible alternatives, not just one state
    - $r: s \times A \rightarrow \R$: reward function
    - Result of an action can be observed only AFTER the action is executed
  - **Stochastic** MDP
    - $f: S \times A \times S' \times A' \rightarrow [0,1]$ defined as transition from a probability distribution
      - $f(s',r|s,a) = Pr(S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a)$
    - From the expression above we can define everything else through marginalization and expected value
      - $f(s'|s,a) = \Sigma_r f(s',r|s,a) \rightarrow$ probability of getting to new state $s'$
      - $r(s,a) = \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a] = \Sigma_r r \Sigma_{s'} f(s',r|s,a)$
        - Expected Value $\mathbb{E}[r|s,a]$ is $\Sigma r f(r|s,a)$
        - The second term is obtained by marginalization

## Goal in RL

- **We aim to find a policy such that the cumulative reward in the long run is maximised**
- How do we define the policy?
  - As a mapping from a state to an action
  - $\pi(a|s): A \times S \rightarrow [0,1]$
  - Given the agent is in a certain state, the action a is executed with a certain probability
- How do we define the cumulative reward?
  - $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t+1} R_{T}$
  - Also called **discounted return**
  - The discount $\gamma$ is needed to give more importance to proximal timesteps
  - We can also define it **recursively**: $G_t = R_{t+1} + \gamma G_{t+1}$
- But this return can actually not be computed on the spot, so we define the **expected return**
  - This is because we have to consider all the possible returns over all possible evolutions from the current state
  - $\mathbb{E}_\pi[G_t|S_t=s] = v_{\pi}(s) \rightarrow$ VALUE FUNCTION
  - $\mathbb{E}_\pi[G_t|S_t=s,A_t=a] = q_{\pi}(s,a) \rightarrow$ ACTION-VALUE FUNCTION
- In the end we need to find $\pi^* = \argmax_\pi v_{\pi}(s)$ $\forall s$
- How? In the next chapters
