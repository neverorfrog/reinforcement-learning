
import numpy as np


def train(environment, trainer, epochs = 5):
    
    rewards = []
    for i in range(epochs):
        print(f"Starting game {i+1}")
        env = environment((3+i),(3+i))
        policy = trainer(env)

        state = env.reset()
        env.render()

        total_reward = 0.
        done = False
        while not done:
            action = policy[state[0],state[1]]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        print("\tTotal Reward:", total_reward)
        rewards.append(total_reward)

    print("Mean Reward: ", np.mean(rewards))
    return env, policy