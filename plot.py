from __future__ import division

import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


plt.rc("axes.spines", top=False, right=False)
sns.set_style("white")
method = "mbieb"
# method = "mbieeb_pc"

## Plot to vary epsilon
every = []
epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
epsilons.reverse()
b = 0.0001
for eps in epsilons:
    arrays = []
    for seed in range(1, 6):
        rewards = np.load('data/{}_s{}_e{}_b{}.npy'.format(method, seed, str(eps).replace('.', '_'), str(b)))
        arrays.append(np.cumsum(rewards))
    every.append(np.vstack(arrays))
sns.tsplot(data=np.stack(every, axis=2), ci="sd", condition=["eps {}".format(str(eps)) for eps in epsilons])

plt.xlabel('Timesteps')
plt.ylabel('Cumulative reward')
plt.legend(loc='upper left', fontsize='medium')
plt.ylim(-20, 260)
plt.title("MBIE-EB on 9-rooms")
# plt.title("MBIE-EB with pseudo-count on 9-rooms")
plt.savefig('plot_eps_{}.png'.format(method), bbox_inches='tight')
plt.show()
plt.close()


## Plot to vary beta
every = []
epsilons = [0.1]
beta = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
for eps in epsilons:
    for b in beta:
        arrays = []
        for seed in range(1, 6):
            rewards = np.load('data/{}_s{}_e{}_b{}.npy'.format(method, seed, str(eps).replace('.', '_'), str(b)))
            arrays.append(np.cumsum(rewards))
        every.append(np.vstack(arrays))
sns.tsplot(data=np.stack(every, axis=2), ci="sd", condition=["beta {}".format(str(b)) for b in beta])

plt.xlabel('Timesteps')
plt.ylabel('Cumulative reward')

plt.legend(loc='upper left', fontsize='medium')
plt.ylim(-20, 260)
plt.title("MBIE-EB on 9-rooms")
# plt.title("MBIE-EB with pseudo-count on 9-rooms")
sns.despine()
plt.savefig('plot_beta_{}.png'.format(method), bbox_inches='tight')
plt.show()
plt.close()