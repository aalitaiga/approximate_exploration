from __future__ import division

import gym
import math
import gym_nine_rooms
import numpy as np
import random
import scipy.signal

from gym_nine_rooms.envs.nine_rooms import RoomSAModel as RoomModel

# Taken from https://github.com/mgbellemare/SkipCTS/blob/master/python/cts/fastmath.py
def log_add(log_x, log_y):
    """Given log x and log y, returns log(x + y)."""
    # Swap variables so log_y is larger.
    if log_x > log_y:
        log_x, log_y = log_y, log_x

    # Use the log(1 + e^p) trick to compute this efficiently
    # If the difference is large enough, this is effectively log y.
    delta = log_y - log_x
    return math.log1p(math.exp(delta)) + log_x if delta <= 50.0 else log_y

def log_minus(log_x, log_y):
    """Given log x and log y, returns log(x - y)."""
    delta = log_y - log_x
    return math.log1p(-math.exp(delta)) + log_x


class Agent(object):
    """ Base class for agents"""

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in self.game_art).astype('float32')
        self.q_values = float('-inf') * np.ones(self.art.shape+(num_actions,))
        self.occupancy = np.zeros(self.art.shape)
        # self.occupancy[self.art == 35] = float('-inf')

    def get_action(self, observation):
        # import ipdb; ipdb.set_trace()
        self.occupancy[observation[0], observation[1]] += 1
        return random.randint(0, self.num_actions-1)

    def get_state_occupancy(self):
        # nb_visits = np.ma.masked_invalid(self.occupancy).sum()
        return self.occupancy / self.occupancy.sum()


class MBIEEB(object):

    def __init__(self, env, epsilon=0.1, beta=0.02, num_actions=4, gamma=0.999, method='mbieeb'):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.num_actions = num_actions
        shape = len(env.env.game_art)
        self.shape = shape
        self.Q = np.zeros((shape*shape, num_actions), dtype=np.float32)
        self.R = np.zeros((shape*shape, num_actions), dtype=np.float32)
        self.P = np.concatenate([np.eye(shape*shape,shape*shape, dtype=np.float32)[:,:,None] for i in range(num_actions)], axis=2).transpose((0,2,1))
        self.N = np.ones((shape*shape,num_actions), dtype=np.float32)
        self.model = RoomModel(3)
        self.mask = self.create_mask().reshape((shape*shape))[:,None]
        self.filter = self.create_room_filter()
        self.pseudo = np.ones((shape, shape, num_actions), dtype=np.float32)
        self.method = method

    def get_pseudo_count(self, room, a):
        log_rho_t = self.model.get_room_prob(room, a)
        log_rho_tp1 = self.model.get_room_prob(room, a, recoding=True)

        log_pseudo = log_rho_t + math.log1p(-math.exp(log_rho_tp1)) - log_minus(log_rho_tp1, log_rho_t)
        return math.exp(log_pseudo)


    def value_iteration_fast(self, gamma=0.999, theta=1e-3):
        shape = self.shape
        transitions = self.P / self.N[:,:,None]

        # MBIE-EB classique or pseudo counts
        if self.method == 'mbieeb':
            rewards = np.expand_dims((self.R / self.N + self.beta / np.sqrt(self.N))*self.mask, axis=2)
        else:
            pseudo_counts = self.compute_pseudo_count().reshape((shape*shape, self.num_actions))
            rewards = np.expand_dims(self.R / self.N + self.beta / np.sqrt(pseudo_counts), axis=2)

        # values = self.Q[:,:,None]
        values = np.zeros(transitions.shape[0], dtype=np.float32)
        delta = np.inf
        # while delta >= theta:
        for _ in range(10):
            q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
            new_values = np.max(q_values, axis=1)
            delta = np.max(np.abs(new_values - values))
            values = new_values

        self.Q = np.squeeze(np.sum(transitions * (rewards + gamma * values), axis=2))
        if np.isnan(np.sum(self.Q)):
            ipdb.set_trace()
        pi = np.argmax(q_values, axis=1)
        return values, pi

    def epsilon_greedy(self, state, epsilon):
        epsilon = epsilon
        rnd = random.uniform(0, 1)

        if rnd <= epsilon:
            return random.randrange(0, self.num_actions)
        else:
            T = self.Q[state[0]*self.shape + state[1]]
            return np.random.choice(np.where(T == T.max())[0])

    def compute_pseudo_count(self):
        for i in range(9):
            for a in range(self.num_actions):
                self.pseudo[self.filter == (i+1), a] = self.get_pseudo_count(i, a)
        return self.pseudo.copy()

    def create_mask(self):
        game_art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in self.env.env.game_art).astype('float32')
        shape = len(game_art)
        mask = np.ones((shape, shape))
        mask[game_art == 35] = 0
        return mask

    def create_room_filter(self):
        array = np.zeros((self.shape, self.shape))
        li = [i for i in range(1, 10)]
        m = 0
        for i, j in zip([1, 8, 15], [8, 15, 21]):
            for k, l in zip([1, 8, 15], [8, 15, 21]):
                array[i:j, k:l] = li[m]
                m += 1
        return array

    def learn_env(self, limit=300):
        epsilon = self.epsilon
        rewards = []
        shape = self.shape
        assert self.Q.sum() == 0
        n = 0
        print("Running MBIEB with eps: {}, beta: {}".format(epsilon, self.beta))

        for i in range(limit):
            if i == 0:
                obs = self.env.reset()
            action = self.epsilon_greedy(obs, epsilon)
            obs_tp1, reward, done, _ = self.env.step(action)
            reward = reward if reward else 0

            # Update the models
            self.model.update(obs, action)
            self.N[obs[0]*shape + obs[1], action] += 1.
            self.P[obs[0]*shape + obs[1], action, obs_tp1[0]*shape + obs_tp1[1]] += 1.
            self.R[obs[0]*shape + obs[1], action] += reward
            obs = obs_tp1.copy()

            if i % 20 == 0:
                self.value_iteration_fast()

            rewards.append(reward)
            if i % 2000 == 0:
                print(np.sum(rewards), i)

            if done:
                self.N[obs_tp1[0]*shape + obs_tp1[1], random.randrange(0, self.num_actions)] += 1.
                obs = self.env.reset()
                n = n + 1
        print("The algorithm solved the environment {} times".format(str(n)))
        rew_array = np.array(rewards)
        return rew_array
