import argparse
import random
import numpy as np
from student import TDLambda_LVFA
import gym
random.seed(1) # do not modify
np.random.seed(1)  # do not modify

def evaluate(fname, env=None, n_episodes=10, max_steps_per_episode=200, render=False):
    env = gym.make('MountainCar-v0')
    if render:
        env = gym.make('MountainCar-v0', render_mode='human')

    agent = TDLambda_LVFA.load(fname)
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        for i in range(max_steps_per_episode):
            action = agent.policy(s)
            
            s_prime, reward, done, _, _ = env.step(action)
            if render: env.render()
            total_reward += reward
            s = s_prime
            if done: break
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))


def train(fname):
    env = gym.make('MountainCar-v0')
    agent = TDLambda_LVFA(env)
    agent.train()
    agent.save(fname)

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', default=None)
    parser.add_argument('-e', '--evaluate', default=None)
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)

    if args.evaluate:
        evaluate(args.evaluate, render=args.render)

    


if __name__ == '__main__':
    main()