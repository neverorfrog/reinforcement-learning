# Model-Based Reinforcement Learning

## Model

We mean anything that an agent can use to predict how the env will respond to
its actions. 

### Why
To simulate or mimic experience

## Planning

Any computational process that takes a model as input and produces or improves a
policy for interacting with the modeled env.

## Recap

Planning and learning in the RL setting share the same modus operandi:
- agent takes a step and collects experience
- computes a backed-up value depending on the target
- updates the estimated value

In planning the experience is collected by interacting with the simulated env,
in learning with the real env (cause we have no model of the env)

# Model-based RL

While interacting with the environment, its model is learnt, using it also to
enhance the policy by approximating better the value function

## Advantages

- Less data needed
- 

