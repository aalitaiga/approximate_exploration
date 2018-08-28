from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import curses
import math
import sys

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
import numpy as np



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

def make_game(game_art, player_sprite):
    """Builds and returns a four-rooms game."""
    return ascii_art.ascii_art_to_game(
        game_art, what_lies_beneath=' ',
        sprites={'P': player_sprite})

class PlayerSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.
    """

    def __init__(self, corner, position, character):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things   # Unused.

        # Apply motion commands.
        if actions == 0:    # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)
        #print(self.position)

        if self.position in [(1, 19), (1, 20), (2, 19), (2, 20)]:
            the_plot.add_reward(1.0)
            the_plot.terminate_episode()

class RoomWorld(gym.Env):
    def __init__(self, game_art):
        self.game_art = game_art
        self.game = ascii_art.ascii_art_to_game(
            game_art, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=len(game_art)-1, shape=(2,))
        self.occupancy = np.zeros((len(self.game_art),len(self.game_art)))
        self.i = 0

    def create_init_prob(self):
        init_prob = np.vstack(np.fromstring(line, dtype=np.uint8) for line in self.game_art).astype('int64')
        init_prob[init_prob == 35] = 0
        init_prob[init_prob == 80] = 1
        init_prob[init_prob == 32] = 1
        return init_prob

    def _step(self, action):
        # Use the sprite position instead of the whole board as an observation
        _, reward, _ = self.game.play(action)
        sprite_position = self.game._sprites_and_drapes['P'].position
        obs = np.array(sprite_position)

        return obs, reward, self.game.game_over, ""

    def _reset(self):
        # Find cleaner way to reset the end, for now just recreate it
        self.game = ascii_art.ascii_art_to_game(
            self.game_art, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        _, reward, _ = self.game.its_showtime()
        observation = np.array(self.game._sprites_and_drapes['P'].position)
        return observation

    def get_room_probs(self):
        total = sum(self.model.count.values())
        return [i / total for i in self.model.count.values()]

    def get_probs(self, pos):
        return [model.get_log_probability(pos) for model in self.model.models]

    @property
    def count(self):
        return self.model.count

    def _render(self, mode="human", close=False):
        # raise NotImplementedError
        pass

class DensityModel(object):
    def __init__(self, n):
        self.count = {(i,j): 1 for i in range(n) for j in range(n)}

    def which_room(self, pos):
        raise NotImplementedError

    def update(self, pos):
        self.count[self.which_room(pos)] += 1

    def get_pseudo_count(self, pos, update_model=True):
        log_rho_t = self.get_log_probability(pos)

        # calculate pseudo counts
        log_rho_tp1 = self.get_log_probability(pos, reco=1)
        prediction_gain = log_rho_tp1 - log_rho_t
        log_pseudo = log_rho_t + math.log1p(-math.exp(log_rho_tp1)) - log_minus(log_rho_tp1, log_rho_t)

        # update models
        self.update_model(pos)

        return log_pseudo, prediction_gain, log_rho_t, log_rho_tp1

    def get_probability(self, pos, reco=0):
        room = self.which_room(pos)
        total = sum(self.count.values())
        return (self.count[room] + reco) / ((total + reco) * 37)

    def get_log_probability(self, pos, recoding=False):
        return math.log(self.get_probability(pos, recoding=recoding))

class UniModel(DensityModel):
    def __init__(self, occupancy):
        self.occupancy = occupancy

    def get_probability(self, pos):
        return (self.occupancy[pos[0], pos[1]]) / (self.occupancy).sum()

    def update(self, pos):
        self.occupancy[pos[0], pos[1]] += 1

def main(game_art, sprite=PlayerSprite):
    # Build a four-rooms game.
    game = make_game(game_art, sprite)

    # Make a CursesUi to play it with.
    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4},
        delay=200
    )

    # Let the game begin!
    ui.play(game)