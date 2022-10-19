from grid_world import NonDeterministicGridWorld
from student import value_iteration, policy_iteration
import argparse
import random
import numpy as np
random.seed(1) # do not modify
np.random.seed(1)  # do not modify

def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    render = args.render

    rewards = []
    for i in range(3):
        print(f"Starting game {i+1}")
        env = NonDeterministicGridWorld(4+i,5+i)
        policy = policy_iteration(env)

        state = env.reset()
        if render: env.render()

        total_reward = 0.
        done = False
        while not done:
            action = policy[state[0],state[1]]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if render: env.render()
        print("\tTotal Reward:", total_reward)
        rewards.append(total_reward)

    print("Mean Reward: ", np.mean(rewards))


if __name__ == '__main__':
    main()