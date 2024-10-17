# Dreamer Definitions

## World Model

### What
- Explicit way to represent an agent's knowledge about its environment
- Summary of agent's experience into a predictive model

### Why
- To use it in place of the environment to learn behaviors

## Latent dynamics models

### What
- Compact state representations (in the latent space)
- Used for world modeling for high-dimensional input spaces

### Why
- Facilitates long-term predictions
- Allows to efficiently predict thousands of compact state sequences in parallel in a single batch, without having
to generate images