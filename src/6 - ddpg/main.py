from agent import DDPG
import gymnasium as gym


if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v4')
    agent = DDPG("ddpg_pendulum", env, max_episodes = 500)
    train = True
    test = True
    if train:
        agent.train()
        agent.save()
    if test:
        env = gym.make('InvertedPendulum-v4', render_mode = 'human')
        agent.load()
        agent.env = env
        agent.evaluate(num_ep = 10)