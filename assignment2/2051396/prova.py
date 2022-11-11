import matplotlib.pyplot as plt
import gym
import numpy as np
# from RBF import *
import random as rd
import math

# the discount factor
discount = 0.995

# The learning rate
alpha = 0.01

# The exploration rate
epsilon = 0.3

# The number of episodes used to evaluate the quality of a policy
n_episodes = 5000

# maximum number of steps of a trajectory
max_steps = 499

# number of grids
n_slices = 20

env = gym.make('MountainCar-v0')
env.max_episode_steps = max_steps + 1

n_a = env.action_space.n
n_s = env.observation_space.shape[0]  # number of features of the states
rd.seed()


def phi_i(s, c_i, sigma_i):
    """
    This function return the value of the Gaussian function
    :param s: value of s
    :param c_i: value of c_i, the center of the feature
    :param sigma_i: value of sigma_i, the length of the feature
    :return: value
    """
    return math.exp(- np.linalg.norm(s - c_i) / sigma_i)


def phi(s, n_slices, disMat):
    res = np.zeros(n_slices)
    for i in range(n_slices):
        res[i] = phi_i(s, disMat[:, i, 2], 1)
    return res


def discr(env, n_slices, n_s):
    """
    This function builds the descretisation matrix of the space of states
    :param env: the environment
    :param n_slices: number of features
    :param n_s: size of the environment's states
    :return: the discretisation matrix
    """
    low = env.observation_space.low
    high = env.observation_space.high
    disMat = np.zeros((n_s, n_slices, 4))
    length = (high - low) / n_slices
    for i in range(n_s):
        for j in range(n_slices):
            if j == n_slices - 1:
                disMat[i, j, 0] = low[i] + length[i] * j  # the lower bound
                disMat[i, j, 1] = high[i]  # the upper bound
                disMat[i, j, 2] = low[i] + length[i] * (j + 0.5)  # the center
                disMat[i, j, 3] = length[i]  # the length
            else:
                disMat[i, j, 0] = low[i] + length[i] * j  # the lower bound
                disMat[i, j, 1] = low[i] + length[i] * (j + 1)  # the upper bound
                disMat[i, j, 2] = low[i] + length[i] * (j + 0.5)  # the center
                disMat[i, j, 3] = length[i]  # the length
    return disMat, length


def findFeature(observation, disMat, env, length):
    """
    This function finds the corresponding feature for an observation
    :param observation: the observation
    :param disMat: the discretisation Matrix
    :return: The correct feature (center, length, number) * 2
    """
    n_s, n_slices, x = disMat.shape
    res = list()
    for i in range(n_s):
        k = int((observation[i] - env.observation_space.low[i]) / length[i])
        try:
            res.append((disMat[i, k, 2], disMat[i, k, 3], k))
        except IndexError:
            res.append((disMat[i, n_slices - 1, 2], disMat[i, n_slices - 1, 3], n_slices - 1))
    return np.array(res)


def indexTabToIndex(indexTab, n_slices, n_s):
    index = 0
    for i in range(n_s):
        index += indexTab[n_s - i - 1] * (n_slices ** i)
    return int(index)


def indexToIndexTab(index, n_slices):
    if index == 0:
        return [0]
    digits = []
    while index:
        digits.append(int(index % n_slices))
        index = index // n_slices
    return digits[::-1]


weight = np.zeros(n_slices)
qlm = np.zeros((n_slices ** n_s, n_a))
qlm[:, 2] = 1

# weight = np.load('MoutainOutput/SARSAMountainCarWeight.npy')
# qlm = np.load('MoutainOutput/SARSAMountainCarMatrix.npy')

disMat, length = discr(env, n_slices, n_s)

print("Let's go !!")

for i_episode in range(n_episodes):

    # print("Episode n° {}".format(i_episode + 1))

    observation = env.reset()
    index = findFeature(observation, disMat, env, length)[:, 2]
    t = rd.random()
    if t < epsilon * 0.25:
        action = 0
    elif epsilon * 0.25 <= t < epsilon * 0.5:
        action = 1
    elif epsilon * 0.5 <= t < epsilon:
        action = 2
    else:
        action = np.argmax(qlm[indexTabToIndex(index, n_slices, n_s), :])

    for i_step in range(max_steps):

        # env.render()
        observationPrime, reward, done, info = env.step(action)

        index = indexTabToIndex(findFeature(observation, disMat, env, length)[:, 2], n_slices, n_s)
        indexPrime = indexTabToIndex(findFeature(observationPrime, disMat, env, length)[:, 2], n_slices, n_s)
        if t < epsilon * 0.25:
            action = 0
        elif epsilon * 0.25 <= t < epsilon * 0.5:
            action = 1
        elif epsilon * 0.5 <= t < epsilon:
            action = 2
        else:
            actionPrime = np.argmax(qlm[indexPrime, :])

        sumAux = (discount * phi(observationPrime, n_slices, disMat) - phi(observation, n_slices, disMat)).dot(weight)
        weight = weight + alpha * (reward + sumAux) * phi(observation, n_slices, disMat)
        qlm[index, action] = phi(observation, n_slices, disMat).dot(weight)

        action = actionPrime
        observation = observationPrime

        achieved = False
        achieved = (observation[0] > 0.5)

        if reward != -1:
            print(reward)

        if achieved:
            print("Episode {} finished after {} timesteps".format(i_episode + 1, i_step + 1))
            break

np.save('MoutainOutput/SARSAMountainCarWeight', weight)
np.save('MoutainOutput/SARSAMountainCarMatrix', qlm)