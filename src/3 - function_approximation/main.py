from agents import TDLambda, Qlearning
import gymnasium as gym
from encoder import RBFFeatureEncoder
from estimator import Estimator

if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode = "rgb_array")
    n_episodes = 90
    encoder = RBFFeatureEncoder(env)
    Q = Estimator(env, encoder)
    agent = Qlearning(env, Q)
    # agent = TDLambda(env, Q)
    agent.train(n_episodes)
    
    testenv = gym.make("MountainCar-v0", render_mode = "human")
    agent.evaluate(testenv, render = True, episodes = 2, max_steps=200)