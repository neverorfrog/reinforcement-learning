import gymnasium as gym
from agent import REINFORCE


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = "rgb_array")     
    agent = REINFORCE("hi", env)
    agent.train()
    
    