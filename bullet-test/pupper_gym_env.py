import math
import time
import gym
import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from pupper import Pupper

TG_PARAMETER = 12

# Register as OpenAI Gym Environment
register(
    id="PupperEnv-v0",
    entry_point='bullet-test.pupper_gym_env:pupperGymEnv',
    max_episode_steps=10000,
)


class pupperGymEnv(gym.Env):
    """THe gym environment for Stanford Pupper.
    It simulates the locomotion of pupper, a quadruped robot.
    The state space include the position, orientation and the action space is the msg for controller.
    The reward function is based on how far the robot walk in 1000 steps and penalizes the yaw drift.
    """

    def __init__(self,
                 render=False,
                 xml_path="bullet-test/pupper_pybullet_out.xml",
                 file_root=pybullet_data.getDataPath(),
                 num_steps_to_log=1000,
                 log_path=None,
                 forward_weight=10.0,
                 rp_wight=2.0,
                 drift_weight=1.0,
                 num_step_to_log=10,
                 motor_kp=0.25,
                 motor_kv=0.5,
                 motor_max_torque=10,
                 hard_reset=False,
                 time_step=0.01,
                 task=1,
                 height_field=False,
                 height_field_no=1,
                 distance_limit=50
                 ):

        # MJCF
        self._xml_path = xml_path
        self._file_root = file_root
        self.task = task

        self._observation = []
        self._true_observation = []
        self.forward_weight = forward_weight
        self.rp_weight = rp_wight
        self.drift_weight = drift_weight
        self.distance_limit = distance_limit
        self.target_position = [20, 0, 0]
        self.target_yaw = np.pi/4
        self.task1_total_reward = 0
        self.task2_total_reward = 0

        self._is_render = render
        self.hard_reset = True
        self.time_step = 1/240  # time_step
        self._env_step_counter = 0
        self._num_step_to_log = num_step_to_log

        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0, 1]

        self._cam_dist = 1.0
        self._cam_yaw1 = -45
        self._cam_yaw2 = -90
        self._cam_pitch = -30

        self._motor_kp = motor_kp
        self._motor_kv = motor_kv
        self._motor_max_torque = motor_max_torque

        self.pb = pybullet
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        self.pb.setGravity(0, 0, -9.8)
        self.seed()
        self.reset()
        self.height_field = height_field
        self.height_field_No = height_field_no
        self.pupper.HeightField(self.height_field, self.height_field_No)
        # observation and action spaces
        observation_high = self.pupper.GetObservationUpperBound()
        observation_low = self.pupper.GetObservationLowerBound()
        self.observation_space = spaces.Box(observation_low, observation_high)
        action_low = self.pupper.GetActionLowerBound()
        action_high = self.pupper.GetActionUpperBound()
        self.action_space = spaces.Box(action_low, action_high)

        self.hard_reset = hard_reset
        self._goal_reached = False

    def reset_cam(self, pos):
        if self.task == 1:
            self.pb.resetDebugVisualizerCamera(
                self._cam_dist, self._cam_yaw1, self._cam_pitch, pos)
        else:
            self.pb.resetDebugVisualizerCamera(
                self._cam_dist, self._cam_yaw2, self._cam_pitch, pos)

    def reset(self):
        self.pb.configureDebugVisualizer(self.pb.COV_ENABLE_RENDERING, 0)
        if self.hard_reset:
            self.pb.resetSimulation()
            self.pb.setGravity(0, 0, -9.8)
            self.pb.setTimeStep(self.time_step)
            # self.ground_id = self.pb.loadURDF(self._file_root + "/plane.urdf")
            self.pupper = Pupper(pybullet_client=self.pb,
                                 height_field=False,
                                 height_field_no=1,
                                 file_root=self._file_root,
                                 motor_kp=self._motor_kp,
                                 motor_kv=self._motor_kv,
                                 motor_max_torque=self._motor_max_torque
                                 )
            self.pupper.reset(reload_mjcf=True)
        else:
            self.pupper.reset(reload_mjcf=False)
        self._env_step_counter = 0

        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0, 1]
        self.task1_total_reward = 0
        self.task2_total_reward = 0
        self.pb.stepSimulation()
        self.reset_cam([0, 0, 0])
        if self._is_render:
            self.pb.configureDebugVisualizer(self.pb.COV_ENABLE_RENDERING, 1)
        return self.pupper.GetObservation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._last_base_position = self.pupper.GetBasePosition()
        self._last_base_orientation = self.pupper.GetBaseOrientation()
        if self._is_render:
            base_pos = self.pupper.GetBasePosition()
            self.reset_cam(base_pos)
        self.pupper.step(action)
        if self.task == 1:
            reward = self.task1_reward()
        elif self.task == 2:
            task2_reward = self.task2_reward()
            self.task2_total_reward += task2_reward
            reward = self.task2_total_reward
        else:
            reward = -1000
            print("Plz set task!")
        done = self._termination()
        self._env_step_counter += 1
        return np.array(self.pupper.GetObservation()), reward, done, {}

    def render(self, mode='human'):
        None

    def is_fallen(self):
        orientation = self.pupper.GetBaseOrientation()
        rot_mat = self.pb.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.55

    def is_reached_goal(self):
        current_pos = self.pupper.GetBasePosition()
        target_pos = self.target_position
        dis = math.sqrt(
            (current_pos[0] - target_pos[0]) ** 2 +
            (current_pos[1] - target_pos[1]) ** 2
        )
        return dis < 0.2

    def _termination(self):
        position = self.pupper.GetBasePosition()
        distance = math.sqrt(position[0] ** 2 + position[1] ** 2)
        return self.is_fallen() or distance > self.distance_limit or self.is_reached_goal()

    def task1_reward(self):
        """task1 for low policy training
            task1 aims to train pupper walking forward
            reward includes forward distance and loss for roll and pitch
        """
        # get reward observation
        pos, orn = self.pupper.Get_Base_PositionAndOrientation()
        done = self.is_reached_goal()
        roll, pitch, yaw = self.pb.getEulerFromQuaternion([orn[0], orn[1], orn[2], orn[3]])

        forward_reward = pos[0]
        # penalty for nonzero roll, pitch
        rp_reward = -(abs(roll) + abs(pitch))
        drift_reward = -abs(pos[1])
        reward = (
                self.drift_weight * drift_reward +
                self.forward_weight * forward_reward +
                self.rp_weight * rp_reward
        )
        if done:
            reward += 1000
        return reward

    def task2_reward(self):
        obs = self.pupper.GetObservation()
        current_pos = obs[0:3]
        target_pos = self.target_position
        last_pos = self._last_base_position
        reward_current = math.sqrt(
            (current_pos[0] - target_pos[0]) ** 2 +
            (current_pos[1] - target_pos[1]) ** 2
        )
        reward_last = math.sqrt(
            (last_pos[0] - target_pos[0]) ** 2 +
            (last_pos[1] - target_pos[1]) ** 2
        )
        self._last_base_position = current_pos
        reward = reward_last - reward_current
        return reward

    @property
    def env_step_counter(self):
        return self._env_step_counter
