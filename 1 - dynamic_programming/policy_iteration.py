from env import *

'''
- Initializtation: all actions are equiprobable in each state
- Policy evaluation: we need to actually understand which policy 
we are folllowing
- Policy improvement: we need to monotonically improve our value
function by applying the Bellman optimality operator
'''

def policy_iteration(env, gamma=0.99, iters=100):

    # Initializing the states
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    # Initializing the policy
    policy = np.zeros(env.num_states, dtype=np.int)
    values = np.zeros(env.num_states, dtype=np.float32)
    

    # Looping for iters iterations to find the optimal policy
    for i in range(iters):

        #Termination condition
        policy_stable = True

        # POLICY EVALUATION: looping on all states until convergence
        eps = 0.1
        while True:

            v_old = values.copy()
            delta = 0

            # Compute the values of the states according to the current policy
            for s in range(env.num_states):
                # Current state
                state = STATES[s]

                if (state == env.end_state).all() or i >= env.max_steps:
                    continue  # if we reach the termination condition, we cannot perform any action

                #with how much proabability i go to the next state
                next_state_prob = env.transition_probabilities(state, policy[s]).flatten()
                
                #EXPECTED UPDATE: value of current policy for the current state, bases on precedent policy on all states
                values[s] = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                #difference from one policy value to the other, until convergence
                delta = max(delta, abs(values[s]-v_old[s]))

            if delta < eps:
                break

        #POLICY IMPROVEMENT: looping on all states
        for s in range(env.num_states):

            # Current state
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue  # if we reach the termination condition, we cannot perform any action

            # Finding the action that maximises the value for the current state
            max_va = values[s]
            best_action = policy[s]

            for a in range(env.num_actions):

                next_state_prob = env.transition_probabilities(state, a).flatten()
                
                #value after selecting current action (q function)
                va = (next_state_prob*(REWARDS + gamma*values)).sum()

                #if, selecting this action, we get a better best value, the policy gets changed
                if va > max_va:
                    best_action = a
                    max_va = va

            if policy[s] != best_action:  # checking for termination
                policy_stable = False
                policy[s] = best_action  # policy improvement

        if policy_stable == True:
            print(f"Optimal policy reached in {i} iterations")
            break

    return policy.reshape(env.height, env.width)