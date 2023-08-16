import TD as td
import gymnasium as gym
from tools.utils import *
from tools.plotting import *

#Training
env = gym.make('CliffWalking-v0')
episodes = 80
board = ProgressBoard(episodes, n = max(episodes / 100, 1))
policy1 = td.qlearning(env, episodes, eps = 0.8, board = board)
policy2 = td.sarsa(env, episodes, eps = 0.8, board = board)
policy3 = td.expected_sarsa(env, episodes, eps = 0.8, board = board)
policy4 = td.double_qlearning(env, episodes, eps = 0.8, board = board)

#Testing
env = gym.make('CliffWalking-v0', render_mode = "human")
test_policy(env, policy1, episodes = 1, max_steps=20)
test_policy(env, policy2, episodes = 1, max_steps=20)
test_policy(env, policy3, episodes = 1, max_steps=20)
test_policy(env, policy4, episodes = 1, max_steps=20)
board.block()

