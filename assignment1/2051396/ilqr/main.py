import gym
from student import CartPole, lqr
import numpy as np
from env import CartPoleEnv
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
	if render:
		env = CartPoleEnv(render_mode='human')
	else:
		env = CartPoleEnv()
	for i in range(3):
		print(f"Starting game {i+1}")
		x,_ = env.reset()

		total_reward = 0
		xf = np.array([0., 0., 0., 0.]) 

		model = CartPole(env, x=x)
		u = np.zeros(1)
		done = False
		while not done:
		    error = x - xf
		    B = model.getB(x=x)
		    A = model.getA(u, x=x)
		    K = lqr(A, B) 
		    u = K@error
		    x, reward, done, _, _ = env.step(u)
		    total_reward += reward
		print("\tTotal Reward:", total_reward)
		rewards.append(total_reward)
	print("Mean Reward: ", np.mean(rewards))
if __name__ == '__main__':
	main()