import gym
import pybullet_data
from pupper import Pupper
import pybullet
from gym.envs.registration import register
from gym import spaces
import numpy as np
from gym.utils import seeding
import math

register(
    id="PupperEnv-1",
    entry_point='bullet-test.high_policy_env:pupperHighGymEnv',
    max_episode_steps=10000,
)


class pupperHighGymEnv(gym.Env):
    def __init__(self,
                 env=None):
        self.duration = 4

        self.target_pos = [3.5, 3.5, 0.3]
        self.last_position = [0, 0, 0]
        self.current_position = []
        action_low, action_high = self.GetActionBound()
        self.action_space = spaces.Box(action_low, action_high)
        observation_bound = self.GetObservationBound()
        self.observation_space = spaces.Box(-observation_bound, observation_bound)
        self.low_policy_env = env
        self.low_policy_env.target_pos

    def reward(self):
        self.current_position = self.low_policy_env.pupper.GetBasePosition()
        r_now = math.sqrt((self.current_position[0] - self.target_pos[0]) ** 2 +
                  (self.current_position[1] - self.target_pos[1]) ** 2 )
        r_pre = math.sqrt((self.last_position[0] - self.target_pos[0]) ** 2 +
                  (self.last_position[1] - self.target_pos[1]) ** 2)
        reward = r_pre - r_now
        self.last_position = self.current_position
        return reward

    def step(self, action):
        target_yaw = action[0]
        self.low_policy_env.target_yaw = target_yaw
        for i in range(self.duration):
            self.low_policy_env.train()

    def GetActionDimension(self):
        action = []
        yaw = 0.0
        # duration = int(1)
        action.append(yaw)
        # action.append(duration)
        return len(action)

    def GetActionBound(self):
        #dt = np.dtype([('yaw', 'f8'), ('duration', 'u8')])
        # low = np.array([(0.5, 1)], dtype=dt)
        low = np.array([0.0] * self.GetActionDimension())
        low[0] = -np.pi
        # low[1] = int(1)
        high = np.array([0.0] * self.GetActionDimension())
        high[0] = np.pi
        # high[1] = int(10)
        return low, high

    def GetObservationDimension(self):
        obs = []
        pos = [0.0, 0.0, 0.33]
        obs.append(pos[0])
        obs.append(pos[1])
        obs.append(pos[2])
        return len(obs)

    def GetObservationBound(self):
        obs_bound = np.array([0.0] * self.GetObservationDimension())
        obs_bound[:] = np.inf
        return obs_bound