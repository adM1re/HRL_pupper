import numpy as np
from transforms3d.euler import euler2mat, quat2euler
from transforms3d.quaternions import qconjugate, quat2axangle
from transforms3d.axangles import axangle2mat
from typing import Any, Tuple
import yaml


# state
class TG_State:
    def __init__(self):
        self.horizontal_velocity = np.array([0.0, 0.0])
        self.yaw_rate = 0.0
        self.height = np.zeros(4)
        self.pitch = 0.0
        self.roll = 0.0
        self.activation = 0

        self.ticks = 0
        self.foot_locations = np.zeros((3, 4))
        self.final_foot_locations = np.zeros((3, 4))
        self.joint_angles = np.zeros((3, 4))

        self.quat_orientation = np.array([1, 0, 0, 0])


class command:
    def __init__(self, height):
        self.horizontal_velocity = np.array([0, 0])
        self.yaw_rate = 0.0
        self.height = height
        self.pitch = 0.0
        self.roll = 0.0

        self.hop_event = False
        self.trot_event = False
        self.activate_event = False

    def __str__(self):
        return "vx: {} vy: {} wz: {} height: {} pitch: {} roll: {} hop_event: {} trot_event: {} act_event: {}".format(
            self.horizontal_velocity[0],
            self.horizontal_velocity[1],
            self.yaw_rate,
            self.height,
            self.pitch,
            self.roll,
            self.hop_event,
            self.trot_event,
            self.activate_event,
        )


# controller config
class Configuration:
    @classmethod
    def from_yaml(cls, yaml_file):
        config = Configuration()
        with open(yaml_file) as f:
            yaml_config = yaml.safe_load(f)
            for k, v in yaml_config.items():
                if k == 'contact_phases':
                    config.contact_phases = np.array(v)
                else:
                    setattr(config, k, v)
        return config

    def __init__(self):
        ################# CONTROLLER BASE COLOR ##############
        self.ps4_color = None
        self.ps4_deactivated_color = None
        #################### COMMANDS ####################
        self.max_x_velocity = 0.0
        self.max_y_velocity = 0.0
        self.max_yaw_rate = 0.0
        self.max_pitch = 0.0 * np.pi / 180.0
        #################### MOVEMENT PARAMS ####################
        self.z_time_constant = 0.0
        self.z_speed = 0.0  # maximum speed [m/s]
        self.pitch_deadband = 0.0
        self.pitch_time_constant = 0.0
        self.max_pitch_rate = 0.0
        self.roll_speed = 0.0  # maximum roll rate [rad/s]
        self.yaw_time_constant = 0.0
        self.max_stance_yaw = 0.0
        self.max_stance_yaw_rate = 0.0
        #################### STANCE ####################
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.x_shift = 0.0
        self.default_z_ref = 0.0
        #################### SWING ######################
        self.z_clearance = 0.0
        self.alpha = (
            0.0  # Ratio between touchdown distance and total horizontal stance movement
        )
        self.beta = (
            0.0  # Ratio between touchdown distance and total horizontal stance movement
        )
        #################### GAIT #######################
        self.dt = 0.0
        self.num_phases = 4
        self.contact_phases = np.ones((4, 4))
        self.overlap_time = (
            0.0  # duration of the phase where all four feet are on the ground
        )
        self.swing_time = (
            0.0  # duration of the phase when only two feet are on the ground
        )

    @property
    def default_stance(self):
        return np.array(
            [
                [
                    self.delta_x + self.x_shift,
                    self.delta_x + self.x_shift,
                    -self.delta_x + self.x_shift,
                    -self.delta_x + self.x_shift,
                ],
                [-self.delta_y, self.delta_y, -self.delta_y, self.delta_y],
                [0, 0, 0, 0],
            ]
        )
    ########################### GAIT ####################
    @property
    def overlap_ticks(self):
        return int(self.overlap_time / self.dt)

    @property
    def swing_ticks(self):
        return int(self.swing_time / self.dt)

    @property
    def stance_ticks(self):
        return 2 * self.overlap_ticks + self.swing_ticks

    @property
    def phase_ticks(self):
        return np.array(
            [self.overlap_ticks, self.swing_ticks, self.overlap_ticks, self.swing_ticks]
        )

    @property
    def phase_length(self):
        return 2 * self.overlap_ticks + 2 * self.swing_ticks


class gait_controller:
    def __init__(self, config):
        self.config = config

    def phase_index(self, ticks):
        """Calculates which part of the gait cycle the robot should be in given the time in ticks.

        Parameters
        ----------
        ticks : int
            Number of timesteps since the program started

        Returns
        -------
        Int
            The index of the gait phase that the robot should be in.
        """
        phase_time = ticks % self.config.phase_length
        phase_sum = 0
        for i in range(self.config.num_phases):
            phase_sum += self.config.phase_ticks[i]
            if phase_time < phase_sum:
                return i
        assert False

    def subphase_ticks(self, ticks):
        """Calculates the number of ticks (timesteps) since the start of the current phase.

        Parameters
        ----------
        ticks : Int
            Number of timesteps since the program started

        Returns
        -------
        Int
            Number of ticks since the start of the current phase.
        """
        phase_time = ticks % self.config.phase_length
        phase_sum = 0
        subphase_ticks = 0
        for i in range(self.config.num_phases):
            phase_sum += self.config.phase_ticks[i]
            if phase_time < phase_sum:
                subphase_ticks = phase_time - phase_sum + self.config.phase_ticks[i]
                return subphase_ticks
        assert False

    def contacts(self, ticks):
        """Calculates which feet should be in contact at the given number of ticks

        Parameters
        ----------
        ticks : Int
            Number of timesteps since the program started.

        Returns
        -------
        numpy array (4,)
            Numpy vector with 0 indicating flight and 1 indicating stance.
        """
        return self.config.contact_phases[:, self.phase_index(ticks)]


class stance_controller:
    def __init__(self, config):
        self.config = config

    def position_delta(self, leg_index, state, command):
        """Calculate the difference between the next desired body location and the current body location

        Parameters
        ----------
        leg_index : float
            Z coordinate of the feet relative to the body.
        state: State
            State object.
        command: Command
            Command object

        Returns
        -------
        (Numpy array (3), Numpy array (3, 3))
            (Position increment, rotation matrix increment)
        """
        z = state.foot_locations[2, leg_index]
        v_xy = np.array(
            [
                -command.horizontal_velocity[0],
                -command.horizontal_velocity[1],
                1.0 / self.config.z_time_constant * (state.height[leg_index] - z),
            ]
        )
        delta_p = v_xy * self.config.dt
        delta_R = euler2mat(0, 0, -command.yaw_rate * self.config.dt)
        return (delta_p, delta_R)

    # TODO: put current foot location into state
    def next_foot_location(self, leg_index, state, command):
        """[summary]

        Parameters
        ----------
        leg_index : [type]
            [description]
        state : [type]
            [description]
        command : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        foot_location = state.foot_locations[:, leg_index]
        (delta_p, delta_R) = self.position_delta(leg_index, state, command)
        incremented_location = delta_R @ foot_location + delta_p

        return incremented_location


class swing_controller:
    def __init__(self, config):
        self.config = config

    def raibert_touchdown_location(self, leg_index, command):
        delta_p_2d = (
            self.config.alpha
            * self.config.stance_ticks
            * self.config.dt
            * command.horizontal_velocity
        )
        delta_p = np.array([delta_p_2d[0], delta_p_2d[1], 0])
        theta = (
            self.config.beta
            * self.config.stance_ticks
            * self.config.dt
            * command.yaw_rate
        )
        R = euler2mat(0, 0, theta)
        return R @ self.config.default_stance[:, leg_index] + delta_p

    def swing_height(self, swing_phase, triangular=True):
        if triangular:
            if swing_phase < 0.5:
                swing_height_ = swing_phase / 0.5 * self.config.z_clearance
            else:
                swing_height_ = self.config.z_clearance * (
                    1 - (swing_phase - 0.5) / 0.5
                )
        return swing_height_

    def next_foot_location(self, swing_prop, leg_index, state, command):
        assert swing_prop >= 0 and swing_prop <= 1
        foot_location = state.foot_locations[:, leg_index]
        swing_height_ = self.swing_height(swing_prop)
        touchdown_location = self.raibert_touchdown_location(leg_index, command)
        time_left = self.config.dt * self.config.swing_ticks * (1.0 - swing_prop)
        v = (touchdown_location - foot_location) / time_left * np.array([1, 1, 0])
        delta_foot_location = v * self.config.dt
        z_vector = np.array([0, 0, swing_height_ + command.height])
        # print("foot_location:{}".format(foot_location * np.array([1, 1, 0])))
        # print("z_vector:{}".format(z_vector))
        # print("delta_foot_location:{}".format(delta_foot_location))
        return foot_location * np.array([1, 1, 0]) + z_vector + delta_foot_location


# controller
class Trajectory_Generator:
    def __init__(self, config):
        self.config = config.from_yaml("controller_config.yaml")
        self.contact_modes = np.zeros(4)
        self.gait_controller = gait_controller(self.config)
        self.stance_controller = stance_controller(self.config)
        self.swing_controller = swing_controller(self.config)

    def step_gait(self,
                  state,
                  command):
        contact_modes = self.gait_controller.contacts(state.ticks)
        new_foot_locations = np.zeros((3, 4))
        for leg_index in range(4):
            contact_mode = contact_modes[leg_index]
            foot_location = state.foot_locations[:, leg_index]
            if contact_mode == 1:
                new_location = self.stance_controller.next_foot_location(
                    leg_index, state, command[leg_index]
                )
            else:
                swing_proportion = (
                        self.gait_controller.subphase_ticks(state.ticks)
                        / self.config.swing_ticks
                )
                new_location = self.swing_controller.next_foot_location(
                    swing_proportion, leg_index, state, command[leg_index]
                )
            # print(new_location)
            new_foot_locations[:, leg_index] = new_location
            return new_foot_locations, contact_modes

    def run(self, state, command):
        state.foot_locations, contact_modes = self.step_gait(state, command)
        state.final_foot_locations = (euler2mat(0.0, 0.0, 0.0) @ state.foot_locations)
        state.ticks += 1
        state.pitch = 0.0
        state.roll = 0.0
        state.height[:] = [command[0].height,
                           command[1].height,
                           command[2].height,
                           command[3].height]
