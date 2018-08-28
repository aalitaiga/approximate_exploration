from __future__ import division

import argparse
import math
import matplotlib
# matplotlib.use('Agg')

import gym
import gym_nine_rooms
from gym_nine_rooms import log_add
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from agent import Agent, MBIEEB

plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='MBIE-EB')
parser.add_argument('seed', default='42')
parser.add_argument('beta', default='0.05', type=float)
args = parser.parse_args()
method = "mbieeb_pc"
# method = "mbieeb"

np.random.seed(int(args.seed))

env = gym.make('NineRooms-v0')
game_art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in env.env.game_art).astype('float32')
shape = len(game_art)
mask = np.ones((shape, shape))
mask[game_art == 35] = 0

# MBIE-EB uses beta 0.005
# MBIE-EB pseudo uses beta 0.01

results = []
agents = []
for eps in [0.2]:
    agent = MBIEEB(env, eps, beta=0.0001)
    rewards = agent.learn_env(limit=20000)
    results.append(rewards)
    agents.append((eps, agent))
    np.save('data_{}/s{}_e{}_b{}.npy'.format(method, args.seed, str(eps).replace('.', '_'), str(args.beta)), rewards)
    env.reset()

    plt.plot(np.cumsum(rewards), label='eps: {}'.format(eps))
plt.xlabel('Timesteps')
plt.ylabel('Cumulative reward')

plt.legend(loc='upper left')
# plt.title("Solve 9-rooms using MBIE-EB with rooom pseudo-count")
plt.title("Solve 9-rooms using classical MBIE-EB")
plt.savefig('data_{}/plot_b{}.png'.format(method, args.beta), bbox_inches='tight')
plt.show()
plt.close()

# for eps, ag in agents:
#     N = ag.N.copy().reshape((shape, shape, 4))
#     M = N.sum(axis=2) * mask
#     # import ipdb; ipdb.set_trace()
#     sns.heatmap(M / M.sum())
#     plt.title("State distribution for eps {}".format(str(eps).replace('.', '_')))
#     plt.savefig('data/{}_heat_e{}_s{}_b{}.png'.format(method, str(eps).replace('.', '_'), args.seed, args.beta), bbox_inches='tight')
#     plt.show()
#     plt.close()