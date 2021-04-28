import numpy as np
from transforms3d.euler import euler2mat
import yaml


class SimHardwareConfig:
    def __init__(self):
        ######################## GEOMETRY ######################
        self.hip_x_offset = 0.10  # front-back distance from center line to leg axis [m]
        self.hip_y_offset = 0.04  # left-right distance from center line to leg plane [m]
        self.lower_link_length = 0.125  # length of lower link [m]
        self.upper_link_length = 0.1235  # length of upper link [m]
        self.abduction_offset = 0.02  # distance from abduction axis to leg [m]

    @property
    def leg_origins(self):
        return np.array(
            [
                [
                    self.hip_x_offset,
                    self.hip_x_offset,
                    -self.hip_x_offset,
                    -self.hip_x_offset,
                ],
                [
                    -self.hip_y_offset,
                    self.hip_y_offset,
                    -self.hip_y_offset,
                    self.hip_y_offset,
                ],
                [0, 0, 0, 0],
            ]
        )

    @property
    def abduction_offsets(self):
        return np.array(
            [
                -self.abduction_offset,
                self.abduction_offset,
                -self.abduction_offset,
                self.abduction_offset,
            ]
        )


def hip_relative_leg_ik(r_body_foot, leg_index, config):
    """Find the joint angles corresponding to the given body-relative foot position for a given leg and configuration

    Parameters
    ----------
    r_body_foot : [type]
        [description]
    leg_index : [type]
        [description]
    config : [type]
        [description]

    Returns
    -------
    numpy array (3)
        Array of corresponding joint angles.
    """
    (x, y, z) = r_body_foot

    # Distance from the leg origin to the foot, projected into the y-z plane
    R_body_foot_yz = (y ** 2 + z ** 2) ** 0.5

    # Distance from the leg's forward/back point of rotation to the foot
    R_hip_foot_yz = (R_body_foot_yz ** 2 + config.abduction_offset ** 2) ** 0.5

    # Interior angle of the right triangle formed in the y-z plane by the leg that is coincident to the ab/adduction axis
    # For feet 2 (front left) and 4 (back left), the abduction offset is positive, for the right feet, the abduction offset is negative.
    arccos_argument = config.abduction_offsets[leg_index] / R_body_foot_yz
    arccos_argument = np.clip(arccos_argument, -0.99, 0.99)
    phi = np.arccos(arccos_argument)

    # Angle of the y-z projection of the hip-to-foot vector, relative to the positive y-axis
    hip_foot_angle = np.arctan2(z, y)

    # Ab/adduction angle, relative to the positive y-axis
    abduction_angle = phi + hip_foot_angle

    # theta: Angle between the tilted negative z-axis and the hip-to-foot vector
    theta = np.arctan2(-x, R_hip_foot_yz)

    # Distance between the hip and foot
    R_hip_foot = (R_hip_foot_yz ** 2 + x ** 2) ** 0.5

    # Angle between the line going from hip to foot and the link L1
    arccos_argument = (config.upper_link_length ** 2 + R_hip_foot ** 2 - config.lower_link_length ** 2) / (
            2 * config.upper_link_length * R_hip_foot
    )
    arccos_argument = np.clip(arccos_argument, -0.99, 0.99)
    trident = np.arccos(arccos_argument)

    # Angle of the first link relative to the tilted negative z axis
    hip_angle = theta + trident

    # Angle between the leg links L1 and L2
    arccos_argument = (config.upper_link_length ** 2 + config.lower_link_length ** 2 - R_hip_foot ** 2) / (
            2 * config.upper_link_length * config.lower_link_length
    )
    arccos_argument = np.clip(arccos_argument, -0.99, 0.99)
    beta = np.arccos(arccos_argument)

    # Angle of the second link relative to the tilted negative z axis
    knee_angle = beta - np.pi

    return np.array([abduction_angle, hip_angle, knee_angle])


def serial_quadruped_inverse_kinematics(r_body_foot, config):
    """Find the joint angles for all twelve DOF correspoinding to the given matrix of body-relative foot positions.

    Parameters
    ----------
    r_body_foot : numpy array (3,4)
        Matrix of the body-frame foot positions. Each column corresponds to a separate foot.
    config : Config object
        Object of robot configuration parameters.

    Returns
    -------
    numpy array (3,4)
        Matrix of corresponding joint angles.
    """
    alpha = np.zeros((3, 4))
    for i in range(4):
        body_offset = config.leg_origins[:, i]
        alpha[:, i] = hip_relative_leg_ik(
            r_body_foot[:, i] - body_offset, i, config
        )
    return alpha