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
import time

INIT_POSITION = [0, 0, 0.4]
INIT_POSITION2 = [0, 0, 0.2]
INIT_ORIENTATION = [0, 0, 0, 1]
INIT_MOTOR_POS = [[-0.2, 0, 1], [1.06, 0, 0], [-1.8, 0, 0],
                  [0.2, 0, -1], [1.06, 0, 0], [-1.8, 0, 0],
                  [-0.2, 0, 0], [1.06, 0, 0], [-1.8, 0, 0],
                  [0.2, 0, 0], [1.06, 0, 0], [-1.8, 0, 0]]
INIT_MOTOR_ANGLE = [-0.1, 0.3, -1.2,
                    0.1, 0.3, -1.2,
                    -0.1, 0.3, -1.2,
                    0.1, 0.3, -1.2]

class Pupper(object):
    def __init__(self,
                 pybullet_client,
                 action_repeat=2,
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

        self.initial_position = INIT_POSITION2
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
                             self.br_command,
                             self.bl_command]
        self.init_state = self.state
        self.x_length = 1
        self.y_length = 0
        self.h_length = -0.12
        self.joint_pos_bound = 2

    def HeightField(self, hf, hf_no):
        if hf:
            if hf_no == 1:
                self.pb.loadURDF(self.file_root + "ground_test1.urdf")
            elif hf_no == 2:
                self.pb.loadURDF(self.file_root + "ground_test2.urdf")

    def ResetPose(self):
        self.pb.resetBasePositionAndOrientation(self.body_id, self.initial_position, self.initial_orientation)

    def reset_joint(self):
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=list(INIT_MOTOR_ANGLE),
            positionGains=[self.motor_kp] * 12,
            velocityGains=[self.motor_kv] * 12,
            forces=[self.motor_max_torque] * 12,
        )
        for _ in range(20):
            pybullet.stepSimulation()
            self.pb.resetBasePositionAndOrientation(self.body_id, INIT_POSITION2, self.initial_orientation)
            time.sleep(6/240)

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
            self.ResetPose()
            self.reset_joint()
        else:
            self.ResetPose()
            self.reset_joint()
        self._step_counter = 0
        time.sleep(1/240)

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
        # joint pos bound
        upper_bound[:] = np.pi
        return upper_bound

    def GetObservationLowerBound(self):
        return -self.GetObservationUpperBound()

    def GetObservationDimension(self):
        return len(self.GetObservation())

    def GetObservation(self):
        observation = []
        pos, orn = self.Get_Base_PositionAndOrientation()
        roll, pitch, yaw = self.pb.getEulerFromQuaternion([orn[0], orn[1], orn[2], orn[3]])
        observation.append(roll)
        observation.append(pitch)
        observation.append(yaw)
        """joint_states = pybullet.getJointStates(self.body_id, self.joint_indices)
        # joint_pos = np.zeros(12)
        # joint_v = np.zeros(12)
        # joint_motor_torque = np.zeros(12)
        for i in range(12):
            observation.append(joint_states[i][0])
            # joint_v[i] = joint_states[i][1]
            # joint_motor_torque[i] = joint_states[i][3]"""
        return observation

    def GetActionUpperBound(self):
        upper_bound = np.array([0.0] * self.GetActionDimension())
        upper_bound[0] = self.x_length
        upper_bound[1] = self.x_length
        upper_bound[2] = self.x_length
        upper_bound[3] = self.x_length
        """upper_bound[4] = self.x_length
        upper_bound[5] = -0.1
        upper_bound[6] = self.x_length
        upper_bound[7] = -0.1"""

        """upper_bound[3] = self.x_length
        upper_bound[4] = self.y_length
        upper_bound[5] = -0.1
        upper_bound[6] = self.x_length
        upper_bound[7] = self.y_length
        upper_bound[8] = -0.1
        upper_bound[9] = self.x_length
        upper_bound[10] = self.y_length
        upper_bound[11] = -0.1"""
        return upper_bound

    def GetActionLowerBound(self):
        low_bound = np.array([0.0] * self.GetActionDimension())
        low_bound[0] = 0.1  # -self.x_length
        low_bound[1] = 0.1  # -self.x_length
        low_bound[2] = 0.1  # -self.x_length
        low_bound[3] = 0.1  # -self.x_length
        """low_bound[4] = -self.x_length
        low_bound[5] = self.h_length
        low_bound[6] = -self.x_length
        low_bound[7] = self.h_length"""
        """low_bound[4] = -self.y_length
        low_bound[5] = self.h_length
        low_bound[6] = -self.x_length
        low_bound[7] = -self.y_length
        low_bound[8] = self.h_length
        low_bound[9] = -self.x_length
        low_bound[10] = -self.y_length
        low_bound[11] = self.h_length"""
        return low_bound

    def GetActionDimension(self):
        action = []
        fr_command = command(self._TG_config.default_z_ref)
        fl_command = command(self._TG_config.default_z_ref)
        bl_command = command(self._TG_config.default_z_ref)
        br_command = command(self._TG_config.default_z_ref)
        action.append(fr_command.horizontal_velocity[0])
        # action.append(fr_command.horizontal_velocity[1])
        # action.append(fr_command.height)
        action.append(fl_command.horizontal_velocity[0])
        # action.append(fl_command.horizontal_velocity[1])
        # action.append(fl_command.height)
        action.append(br_command.horizontal_velocity[0])
        # action.append(br_command.horizontal_velocity[1])
        # action.append(br_command.height)
        action.append(bl_command.horizontal_velocity[0])
        # action.append(bl_command.horizontal_velocity[1])
        # action.append(bl_command.height)
        return len(action)

    def transformAction2Command(self, action):
        self.fr_command.horizontal_velocity[0] = action[0]
        self.fr_command.height = -0.15  # action[1]
        self.fl_command.horizontal_velocity[0] = action[1]
        self.fl_command.height = -0.15  # action[1]
        self.br_command.horizontal_velocity[0] = action[2]
        self.br_command.height = -0.15  # action[1]
        self.bl_command.horizontal_velocity[0] = action[3]
        self.bl_command.height = -0.15  # action[1]
        return [self.fr_command,
                self.fl_command,
                self.br_command,
                self.bl_command]

    def transformAction2Command2(self, action):
        # using for debug locomotion
        self.fr_command.horizontal_velocity = action[0:2]
        self.fr_command.height = action[2]
        return self.fr_command

    def foot_position2motor_angle(self, foot_positions):
        serial_joint_angles = kinematics.serial_quadruped_inverse_kinematics(
            foot_positions, self.sim_hardware_config
        )
        # print(serial_joint_angles)
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=list(serial_joint_angles.T.reshape(12)),
            positionGains=[self.motor_kp] * 12,
            velocityGains=[self.motor_kv] * 12,
            forces=[self.motor_max_torque] * 12,
        )

    def action_limit(self, action):
        action[0] = np.clip(action[0], -self.x_length, self.x_length)
        action[1] = np.clip(action[1], -self.x_length, self.x_length)
        action[2] = np.clip(action[2], -self.x_length, self.x_length)
        action[3] = np.clip(action[3], -self.x_length, self.x_length)
        for i in range(3):
            if abs(action[i]) < 0.001:
                action[i] = action[i] * 1000
            elif abs(action[i]) < 0.01:
                action[i] = action[i] * 100
            elif abs(action[i]) < 0.1:
                action[i] = action[i] * 10
        # print(action[0])
        """action[4] = np.clip(action[4], -self.x_length, self.x_length)
        action[5] = np.clip(action[5], self.h_length, -0.1)
        action[6] = np.clip(action[6], -self.x_length, self.x_length)
        action[7] = np.clip(action[7], self.h_length, -0.1)"""

        """action[3] = np.clip(action[3], -self.x_length, self.x_length)
        action[4] = 0  # np.clip(action[4], self.y_length, -self.y_length)
        action[5] = np.clip(action[5], self.h_length, -0.1)
        action[6] = np.clip(action[6], -self.x_length, self.x_length)
        action[7] = 0  # np.clip(action[7], self.y_length, -self.y_length)
        action[8] = np.clip(action[8], self.h_length, -0.1)
        action[9] = np.clip(action[9], -self.x_length, self.x_length)
        action[10] = 0  # np.clip(action[10], self.y_length, -self.y_length)
        action[11] = np.clip(action[11], self.h_length, -0.1)"""
        return action

    def ApplyAction(self, action):
        # action include :  [[forward_step_length, lateral_length, single_height]*4]
        # print(action[0])
        action = self.action_limit(action)
        self.command = self.transformAction2Command(action)
        self.TG.run(self.state, self.command)
        self.foot_position2motor_angle(self.state.final_foot_locations)

    def ApplyAction2(self, action):
        # using for debug locomotion
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=list(action),
            positionGains=[self.motor_kp] * 12,
            velocityGains=[self.motor_kv] * 12,
            forces=[self.motor_max_torque] * 12,
        )

    def step(self, action):
        for _ in range(self._action_repeat):
            # print("step")
            self.ApplyAction(action)
            # self.ApplyAction2(action)
            time.sleep(1/240)
            self.pb.stepSimulation()
            self._step_counter += 1
