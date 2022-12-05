from DDQN import DDQN_agent
from Experience_replay_buffer import Experience_replay_buffer
import gym

def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    rew_threshold = 400
    buffer = ERB()
    agent = DDQN(env, rew_threshold, buffer)
    agent.train()

    eval_env = gym.make("CartPole-v1", render_mode="human")
    agent.evaluate(eval_env)

if __name__ == '__main__':
    main()

