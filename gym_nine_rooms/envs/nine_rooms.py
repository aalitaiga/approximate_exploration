from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

try:
    from gym_nine_rooms.envs.base import make_game, main, RoomWorld, PlayerSprite, DensityModel
except ImportError:
    from base import make_game, main, RoomWorld, PlayerSprite, DensityModel


GAME_ART = [
    '######################',
    '#      #      #      #',
    '#      #      #      #',
    '#      #             #',
    '#             #      #',
    '#      #      #      #',
    '#      #      #      #',
    '#### ##### ####### ###',
    '#      #      #      #',
    '#      #      #      #',
    '#      #             #',
    '#             #      #',
    '#      #      #      #',
    '#      #      #      #',
    '### ####### ##### ####',
    '#      #      #      #',
    '#      #      #      #',
    '#      #             #',
    '#             #      #',
    '#      #      #      #',
    '#P     #      #      #',
    '######################',
]

class RoomModel(DensityModel):
    def __init__(self):
        super(RoomModel, self).__init__(3)

    def which_room(self, pos):
        row = (pos[0] - 1) // 7
        col = (pos[1] - 1) // 7
        return row, col


    def get_probability(self, pos, reco=0):
        room = self.which_room(pos)
        total = sum(self.count.values())
        if room[0] <= 1 and room[1] <= 1:
            divid = 38
        elif room[0] == 2 and room[1] == 2:
            divid = 36
        else:
            divid = 37
        return (self.count[room] + reco) / ((total + reco) * divid)

class RoomSAModel(DensityModel):

    def __init__(self, n):
        self.count = {(i,j, a): 1 for i in range(n) for j in range(n) for a in range(4)}

    def which_room(self, pos):
        row = (pos[0] - 1) // 7
        col = (pos[1] - 1) // 7
        return row, col

    def get_room_prob(self, coord, a, reco=0):
        room = (coord // 3, coord % 3, a)
        total = sum(self.count.values())
        if room[0] <= 1 and room[1] <= 1:
            divid = 38
        elif room[0] == 2 and room[1] == 2:
            divid = 36
        else:
            divid = 37
        return math.log((self.count[room] + reco) / ((total + reco) * divid))

    def update(self, pos, a):
        room = self.which_room(pos)
        self.count[room[0], room[1], a] += 1

class NineRooms(RoomWorld):
    def __init__(self):
        super(NineRooms, self).__init__(GAME_ART)
        self.model = RoomModel()

    @property
    def count(self):
        return self.model.count


if __name__ == '__main__':
    main(GAME_ART)
