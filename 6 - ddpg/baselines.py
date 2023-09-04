import gymnasium as gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('FetchReach-v2')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", 
            env, 
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render()