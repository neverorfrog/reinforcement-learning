from agents import ControlDeepQAgent
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = "rgb_array")     
    # env = gym.make("MountainCar-v0")   
    n = 200
    
    agent = ControlDeepQAgent("hi", env)
    agent.train()
    
    # testenv = gym.make("MountainCar-v0", render_mode = "human")
    testenv = gym.make('CartPole-v1', render_mode = "human")
    agent.evaluate(testenv, render = True, episodes = 10, max_steps=200)
