from gym.envs.registration import register
from gym_nine_rooms.envs import log_add


register(
    id='NineRooms-v0',
    entry_point='gym_nine_rooms.envs:NineRooms',
    max_episode_steps=20000,
)