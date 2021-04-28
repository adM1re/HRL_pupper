import pybullet
import pybullet_data
import numpy as np
import collections
import copy
import math
from trajectory_generator import Trajectory_Generator as TG
from trajectory_generator import command
from trajectory_generator import Configuration
from trajectory_generator import TG_State
import kinematics
from kinematics import SimHardwareConfig

INIT_POSITION = [0, 0, 0.3]
INIT_POSITION2 = [0, 0, 0.17]
INIT_ORIENTATION = [0, 0, 0, 1]
INIT_JOINT_STATES = [[-0.204, 0.013, [0, 0, 0, 0, 0, 0], -0.670], [1.072, 0.043, [0, 0, 0, 0, 0, 0], 0.293],
                     [-1.873, -0.114, [0, 0, 0, 0, 0, 0], 0.760], [0.204, -0.056, [0, 0, 0, 0, 0, 0], 0.565],
                     [1.071, -0.012, [0, 0, 0, 0, 0, 0], 0.516], [-1.877, -0.036, [0, 0, 0, 0, 0, 0], 0.700],
                     [-0.212, 0.006, [0, 0, 0, 0, 0, 0], 0.622], [1.074, 0.009, [0, 0, 0, 0, 0, 0], -0.498],
                     [-1.854, 0.168, [0, 0, 0, 0, 0, 0], -0.158], [0.212, 0.057, [0, 0, 0, 0, 0, 0], -0.580],
                     [1.073, 0.019, [0, 0, 0, 0, 0, 0], -0.488], [-1.855, 0.065, [0, 0, 0, 0, 0, 0], -0.119]]
INIT_JOINT_POS = [[-0.204, 0.013, -0.670], [1.072, 0.043, 0.293],
                     [-1.873, -0.114,  0.760], [0.204, -0.056, 0.565],
                     [1.071, -0.012, 0.516], [-1.877, -0.036, 0.700],
                     [-0.212, 0.006, 0.622], [1.074, 0.009,  -0.498],
                     [-1.854, 0.168, -0.158], [0.212, 0.057, -0.580],
                     [1.073, 0.019, -0.488], [-1.855, 0.065, -0.119]]

class Pupper(object):
    def __init__(self,
                 pybullet_client,
                 action_repeat=1,
                 time_step=0.01,
                 motor_kp=0,
                 motor_kv=0,
                 motor_max_torque=0,
                 np_random=np.random,
                 height_field=False,
                 height_field_no=1,
                 contacts=True,
                 file_root=None):

        self.pb = pybullet_client
        self.file_root = file_root
        self._pupper_all_body_ids = None
        self.body_id = None
        self.joint_indices = None
        self.motor_kp = motor_kp
        self.motor_kv = motor_kv
        self.motor_max_torque = motor_max_torque

        self.initial_position = INIT_POSITION
        self.initial_orientation = INIT_ORIENTATION

        self._action_repeat = action_repeat
        self.num_motors = 12
        self.num_legs = int(self.num_motors / 3)
        self._motor_direction = np.ones(self.num_motors)
        self._observed_motor_torques = np.zeros(self.num_motors)
        self._applied_motor_torques = np.zeros(self.num_motors)
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []
        self.np_random = np_random
        self._step_counter = None
        self.time_step = time_step
        self.contacts = contacts
        self.initial_pose = None
        # enable rough terrain or not
        self._height_field = height_field
        self._height_field_No = height_field_no
        #
        self.sim_hardware_config = SimHardwareConfig()
        self._TG_config = Configuration.from_yaml("controller_config.yaml")
        self.TG = TG(self._TG_config)
        self.state = TG_State()
        self.fr_command = command(self._TG_config.default_z_ref)
        self.fl_command = command(self._TG_config.default_z_ref)
        self.bl_command = command(self._TG_config.default_z_ref)
        self.br_command = command(self._TG_config.default_z_ref)
        self.command = np.array(4)
        self.init_command = [self.fr_command,
                             self.fl_command,
                             self.bl_command,
                             self.br_command]
        self.init_state = self.state

    def HeightField(self, hf, hf_no):
        if hf:
            if hf_no == 1:
                self.pb.loadURDF(self.file_root + "ground_test1.urdf")
            elif hf_no == 2:
                self.pb.loadURDF(self.file_root + "ground_test2.urdf")

    def ResetPose(self, body_ids, init_pos, init_orn):
        self.pb.resetBasePositionAndOrientation(body_ids, init_pos, init_orn)
        # self.TG.run(self.init_state, self.init_command)
        # self.pb.resetBasePositionAndOrientation(body_ids, init_pos, init_orn)

    def reset(self,
              reload_mjcf=True,
              default_motor_angles=None,
              reset_time=2.0):

        """Reset the pupper to its initial states.
        Args:
        reload_mjcf:
            Whether to reload the mjcf file. If not, Reset() just place the pupper back to its starting position.
        default_motor_angles:
            The default motor angles. If it is None, pupper will hold a default pose for 100 steps.
            In torque control mode, the phase of holding the default pose is skipped.
        reset_time:
            The duration (in seconds) to hold the default motor angles.
            If reset_time <= 0 or in torque control mode, the phase of holding the default pose is skipped.
        """
        #
        if reload_mjcf:
            self._pupper_all_body_ids = self.pb.loadMJCF("pupper_pybullet_out.xml")
            self.body_id = self._pupper_all_body_ids[1]
            self.numjoints = pybullet.getNumJoints(self.body_id)
            self.joint_indices = list(range(0, 24, 2))
            # print(self.body_id)
            self.ResetPose(self.body_id, self.initial_position, self.initial_orientation)
        else:
            self.ResetPose(self.body_id, self.initial_position, self.initial_orientation)
        self._step_counter = 0

        """self._observation_history.clear()
        if reset_time > 0.0 and default_motor_angles is not None:
            for _ in range(100):
                self.ApplyAction(self.initial_pose)
                self.pb.stepSimulation()
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self.ApplyAction(default_motor_angles)
                self.pb.stepSimulation()"""

    def Get_Base_PositionAndOrientation(self):
        position, orientation = pybullet.getBasePositionAndOrientation(self.body_id)
        return position, orientation

    def GetBasePosition(self):
        position, _ = self.pb.getBasePositionAndOrientation(self.body_id)
        return position

    def GetBaseOrientation(self):
        _, orientation = self.pb.getBasePositionAndOrientation(self.body_id)
        return orientation

    def GetObservationUpperBound(self):
        """Get the upper bound of the observation.
                Returns:
                  The upper bound of an observation. See GetObservation() for the details
                    of each element of an observation.
                  NOTE: Changed just like GetObservation()
                """
        upper_bound = np.array([0.0] * self.GetObservationDimension())
        # x , y , z
        upper_bound[0:3] = np.inf
        # roll pitch yaw
        upper_bound[3:] = 2.0 * np.pi
        return upper_bound

    def GetObservationLowerBound(self):
        return -self.GetObservationUpperBound()

    def GetObservationDimension(self):
        return len(self.GetObservation())

    def GetObservation(self):
        observation = []
        pos, orn = self.Get_Base_PositionAndOrientation()
        roll, pitch, yaw = self.pb.getEulerFromQuaternion([orn[0], orn[1], orn[2], orn[3]])
        observation.append(pos[0])
        observation.append(pos[1])
        observation.append(pos[2])
        observation.append(roll)
        observation.append(pitch)
        observation.append(yaw)
        return observation

    def GetActionUpperBound(self):
        upper_bound = np.array([0.0] * self.GetActionDimension())
        upper_bound[0:2] = 0.5
        upper_bound[3:5] = 0.5
        upper_bound[6:8] = 0.5
        upper_bound[9:11] = 0.5
        upper_bound[2] = 0.3
        upper_bound[5] = 0.3
        upper_bound[8] = 0.3
        upper_bound[11] = 0.3
        return upper_bound

    def GetActionLowerBound(self):
        low_bound = np.array([0.0] * self.GetActionDimension())
        low_bound[0:2] = -0.5
        low_bound[3:5] = -0.5
        low_bound[6:8] = -0.5
        low_bound[9:11] = -0.5
        # low_bound[2][5][8][11] = -0.0
        return low_bound

    def GetActionDimension(self):
        action = []
        fr_command = command(self._TG_config.default_z_ref)
        fl_command = command(self._TG_config.default_z_ref)
        bl_command = command(self._TG_config.default_z_ref)
        br_command = command(self._TG_config.default_z_ref)
        action.append(fr_command.horizontal_velocity[0])
        action.append(fr_command.horizontal_velocity[1])
        action.append(fr_command.height)
        action.append(fl_command.horizontal_velocity[0])
        action.append(fl_command.horizontal_velocity[1])
        action.append(fl_command.height)
        action.append(bl_command.horizontal_velocity[0])
        action.append(bl_command.horizontal_velocity[1])
        action.append(bl_command.height)
        action.append(br_command.horizontal_velocity[0])
        action.append(br_command.horizontal_velocity[1])
        action.append(br_command.height)
        return len(action)

    def transformAction2Command(self, action):
        self.fr_command.horizontal_velocity = action[0:2]
        self.fr_command.height = action[2]
        self.fl_command.horizontal_velocity = action[3:5]
        self.fl_command.height = action[5]
        self.bl_command.horizontal_velocity = action[6:8]
        self.bl_command.height = action[8]
        self.br_command.horizontal_velocity = action[9:11]
        self.br_command.height = action[11]
        return [self.fr_command,
                self.fl_command,
                self.bl_command,
                self.br_command]

    def foot_position2motor_angle(self, foot_positions):
        serial_joint_angles = kinematics.serial_quadruped_inverse_kinematics(
            foot_positions, self.sim_hardware_config
        )
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=list(serial_joint_angles.T.reshape(12)),
            positionGains=[self.motor_kp] * 12,
            velocityGains=[self.motor_kv] * 12,
            forces=[self.motor_max_torque] * 12,
        )

    def ApplyAction(self, action):
        # action include :  [[forward_step_length, lateral_length, single_height]*4]
        self.command = self.transformAction2Command(action)
        self.TG.run(self.state, self.command)
        self.foot_position2motor_angle(self.state.final_foot_locations)

    def step(self, action):
        for _ in range(self._action_repeat):
            # print("step")
            self.ApplyAction(action)
            self.pb.stepSimulation()
            self._step_counter += 1

    def GetTimeSinceReset(self):
        return self._step_counter * self.time_step

    """def RealisticObservation(self):
        """"""Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """"""
        self._observation_history.appendleft(self.GetObservation())
        self._control_observation = self._GetDelayedObservation(
            self._control_latency)
        self._control_observation = self._AddSensorNoise(
            self._control_observation, self._observation_noise_stdev)
        return self._control_observation
"""