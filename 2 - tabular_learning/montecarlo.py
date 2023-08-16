import numpy as np

'''
Unknown model
- In dynamic programming we had the model, here we don't
- It means that in the Bellman equation (page 59) we don't have the transition probabilities and the value for s'
- What we do is sampling the return instead of explicitly computing it

How do we sample the return?
- We do a big number of episodes, generating the entire episode
- After that, we iterate over the state-action pairs backwards in time
- Every time we encounter a state-action pair, we sample its return by summing the rewards
- For eache state-action pair, there is an entry in a tensor of shape (#ep, #states, #actions)
- We then estimate the value of a state-action pair by averaging along the first dimension of the tensor
'''

def montecarlo(env, gamma = 0.99, epochs = 3, steps = 100):
    
    # Initializing the states
    states = np.zeros((env.num_states, 2), dtype=np.uint8)
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            states[i] = state
            i += 1

    # Initializing the policy
    policy = np.random.randint(0,7,(env.height,env.width), dtype=np.int)
    Q = np.zeros((env.num_states, env.num_actions), dtype=np.float32)
    
    for epoch in range(epochs):
        #Generate a complete episode
        episode = []
        new_state = env.reset()
        for step in range(steps):
            state = new_state
            action = policy[state[0],state[1]]
            new_state, reward, done, _ = env.step(action)
            present = False
            for earlier_step in episode:
                if (earlier_step[0] == state).all() and earlier_step[1] == action:
                    present = True
                    break
            if not present:
                episode.append([state,action,reward])
            env.render()
            if(done): break
            
        #Loop over the steps to find a good policy
        G = 0
        episode = episode[::-1] #reverse
        for step in range(len(episode)):
            state = episode[step][0]
            action = episode[step][1]
            reward = episode[step][2]
            G = gamma*G + reward
            Q[state,action] = Q[state,action] + 1/(step+1)*(G - Q[state,action]) 
        
        print(f"Q: {Q} for epoch {epoch}")

    return Q
