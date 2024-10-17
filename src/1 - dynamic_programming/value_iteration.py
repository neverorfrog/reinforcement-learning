from gridworld import *
    
'''
- What we do in value iteration is that we continuously apply the Bellman optimality operator
- But without first evaluating the value function
- This means we are basically just doing one policy evaluation step
- After that we improve the policy with bellman optimalty
'''
    
def value_iteration(env):
    gamma = 0.9
    iters = 100

    # initialize values
    values = np.zeros((env.num_states), dtype=float)
    policy = np.zeros((env.num_states), dtype=int)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(iters):
        v_old = values.copy()
        
        #Termination condition
        policy_stable = True
        
        for s in range(env.num_states):
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue  # if we reach the termination condition, we cannot perform any action

            max_va = -np.inf
            best_a = 0
            for a in range(env.num_actions):

                next_state_prob = env.transition_probabilities(state, a).flatten()
                va = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                if va > max_va:
                    max_va = va
                    best_a = a
                    
            values[s] = max_va
            if policy[s] != best_a:  # checking for termination
                policy_stable = False
                policy[s] = best_a  # policy improvement
                
        if policy_stable == True:
            print(f"Optimal policy reached in {i} iterations")
            break
    
    return policy.reshape((env.height, env.width))