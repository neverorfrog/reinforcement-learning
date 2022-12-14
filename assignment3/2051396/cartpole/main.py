import argparse
import random
import numpy as np
from student import Policy
import gym

def evaluate(env=None, n_episodes=5, render=True):
    agent = Policy()
    agent.load()

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        
        steps = 0
        
        while not done:
            action = agent.act(s)
            
            s, reward, done, truncated, info = env.step(action)
            if render: env.render()
            total_reward += reward
            if done or truncated: break
            
            steps = steps + 1
            # print("STEPS={}".format(steps))
        
        rewards.append(total_reward)
        print('Total Reward:', total_reward)
        
    print('Mean Reward:', np.mean(rewards))


def train():
    agent = Policy()
    agent.train()
    agent.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    main()
