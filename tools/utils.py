import numpy as np


def test(env, policy, episodes = 5):
    
    rewards = []
    for i in range(episodes):
        print(f"Starting game {i+1}")

        state = env.reset()[0]
        total_reward = 0.
        terminated = False
        truncated = False
        while not terminated or truncated:
            action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print("\tTotal Reward:", total_reward)
        rewards.append(total_reward)

    print("Mean Reward: ", np.mean(rewards))