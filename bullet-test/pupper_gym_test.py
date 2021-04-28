import math
import time
import gym
import numpy as np
import pybullet
import pybullet_data
import pupper
from utils import bullet_client as bc
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import pupper_gym_env

DEFAULT_MJCF_VERSION = "default"
pupper_MJCF_VERSION_MAP = {DEFAULT_MJCF_VERSION: pupper.Pupper}
NUM_MOTORS = 12
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
NUM_SIMULATION_ITERATION_STEPS = 1000
OBSERVATION_EPS = 0.01

# Register as OpenAI Gym Environment
register(
    id="PupperEnvTest-v0",
    entry_point='pupper_gym_test:pupperGymTest',
    max_episode_steps=1000,
)


class pupperGymEnvTest(gym.Env):
    """THe gym environment for Stanford Pupper.
        It simulates the locomotion of pupper, a quadruped robot.
        The state space include the position, orientation and the action space is the msg for controller.
        The reward function is based on how far the robot walk in 1000 steps and penalizes the yaw drift.
        """

    def __init__(self,
                 action_repeat,
                 distance_weight=0.5,
                 drift_weight=0.5,
                 mjcf_version=None,
                 hard_rest=True,
                 render=True,
                 control_time_step=None,
                 log_path=None):
        self._mjcf_version = mjcf_version
        self._observation_space = []
        self._action_space = []
        self._action_dim = 1
        self._is_render = render
        self._xml_path = "pupper_pybullet_out.xml"
        # self._ground_id = []
        self._robot_id = []
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0, 1]
        self._distance_weight = distance_weight
        self._drift_weight = drift_weight

        # set log path
        self._log_path = log_path

        # NUM ITERS
        self._time_step = 0.01
        self._action_repeat = action_repeat
        self._num_bullet_solver_iterations = 300
        # PD control needs smaller time step for stability
        if control_time_step is not None:
            self.control_time_step = control_time_step
        else:
            self.control_time_step = self._time_step * self._action_repeat

        # 是否渲染
        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        # 模型文件
        if self._mjcf_version is None:
            self._mjcf_version = DEFAULT_MJCF_VERSION
        # 完全重置
        self._hard_rest = hard_rest
        self.reset()
        observation_high = (self.spot.GetObservationUpperBound() + OBSERVATION_EPS)
        observation_low = (self.spot.GetObservationLowerBound() - OBSERVATION_EPS)
        action_dim = NUM_MOTORS
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(observation_low, observation_high)
        self.goal_reached = False

    def set_env_randomizer(self, env_randomizer):
        pass

    def step(self, action):
        pybullet.stepSimulation()
        time.sleep(1 / 240)
        reward = 0
        done = False
        return reward, done

    def reset(self,
              initial_motor_angles=None,  # 初始电机角度
              reset_duration=2.0):  # 重置时间
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 0)  # 关闭渲染
        if self._hard_rest:  # 是否强制重置
            self._pybullet_client.resetSimulation()
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.setGravity(0, 0, -9.81) # 设置重力
            self._robot_all_body_id = pybullet.loadMJCF(self._xml_path)  # 获取机器人编号
            self._robot_id = self._robot_all_body_id[0]  # 机器人主体编号
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 1)  # 开启渲染
            # if self._mjcf_version not in pupper_MJCF_VERSION_MAP:
            #   raise ValueError("%s is not a supported mjcf version." % self._mjcf_version)
            # else:
            #    self.pupper = (pupper_MJCF_VERSION_MAP[self._mjcf_version](
            #
            #    ))
        self.pupper.reset(reload_mjcf=False,
                          default_motor_angles=initial_motor_angles,
                          reset_time=reset_duration)
        self._env_step_counter = 0
        return self._get_observation()

    def seed(self, seed=None):
        pass

    def _transform_action_to_motor_commond(self, action):
        pass

    def render(self, mode='human'):
        pass

    # get pupper motor angles/velocity/torque
    def get_pupper_motor_angles(self):
        return np.array(
            self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:
                              MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS])

    def get_pupper_motor_velocity(self):
        return np.array(
            self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:
                              MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

    def get_pupper_motor_torque(self):
        return np.array(
            self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:
                              MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

    def _termination(self):
        return

    def _reward(self):
        reward = 0
        return reward

    def _get_observation(self):
        self._observation = self.pupper.getObservation()
        return self._observation

    def _get_realistic_observation(self):
        self._observation = self.pupper.RealisticObservation()
        return self._observation

    def set_time_step(self, control_step, simulation_step=0.001):
        if control_step < simulation_step:
            raise ValueError(
                "Control step time should be bigger than simulation step ")
        self.control_time_step = control_step
        self._time_step = simulation_step
        self._action_repeat = int(round(control_step/simulation_step))
        self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS /
                                              self._action_repeat)


def main():
    env = pupper_gym_env.pupperGymEnv(render=True,  task=1,   height_field=0)
    env.reset()

    while True:
        pybullet.stepSimulation()
        # env.reset()


if __name__ == '__main__':
    main()
