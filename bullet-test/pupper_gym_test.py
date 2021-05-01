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
import trajectory_generator

INIT_MOTOR_ANGLE = [-0.1, 0.3, -1.2,
                    0.1, 0.3, -1.2,
                    -0.1, 0.3, -1.2,
                    0.1, 0.3, -1.2]
INIT_MOTOR_ANGLE2 = [0.2, -2, 2,
                     -0.2, 2, -2,
                     0.2, -2, 2,
                     -0.2, 2, -2]

def main():
    env = pupper_gym_env.pupperGymEnv(render=False, task=2, height_field=1, hard_reset=False)
    print(env.reset())
    action = [0.4, 0.7,
              0.4, -0.2,
              0.4, -0.2,
              0.4, -0.2]
    action1 = [0.5, 0, 0.5]
    action2 = [0.5, 0, 0.5]
    action3 = [0.5, 0, 0.5]
    action4 = [0.5, 0, 0.5]
    last_loop = 0
    env.pupper.ResetPose()
    """joint_states = pybullet.getJointStates(env.pupper.body_id, env.pupper.joint_indices)
    joint_pos = np.zeros(12)
    joint_v = np.zeros(12)
    joint_motor_torque = np.zeros(12)
    for i in range(12):
        joint_pos[i] = joint_states[i][0]
        joint_v[i] = joint_states[i][1]
        joint_motor_torque[i] = joint_states[i][3]
    print(joint_pos)
    print(joint_v)
    print(joint_motor_torque)"""
    # Step the controller forward by dt
    # env.step(action)
    # file_root = pybullet_data.getDataPath()
    # pupper_ground = pybullet.loadURDF(file_root + "/ground_test.urdf")
    # print("pupper_ground:")
    # print(pupper_ground)
    # pybullet.resetBasePositionAndOrientation(pupper_ground, [9.4, 0, -0.2], [1, 1, 1, 1])
    """pybullet.setJointMotorControlArray(
        bodyUniqueId=env.pupper.body_id,
        jointIndices=env.pupper.joint_indices,
        controlMode=pybullet.POSITION_CONTROL,
        targetPositions=list(INIT_MOTOR_ANGLE),
        positionGains=[env.pupper.motor_kp] * 12,
        velocityGains=[env.pupper.motor_kv] * 12,
        forces=[env.pupper.motor_max_torque] * 12,
    )"""
    for _ in range(30):
        env.pupper.ResetPose()
    while True:
        now = time.time()
        if now - last_loop >= 0.01:
            # Check if we should transition to "deactivated"
            # env.pupper.ResetPose()
            # pybullet.resetBasePositionAndOrientation(env.pupper.body_id, [2, 1, 2], env.pupper.initial_orientation)

            # Step the controller forward by dt
            # env.step(action)
            # env.pupper.TG.run(state=env.pupper.state, command=env.pupper.command)
            # env.pupper.foot_position2motor_angle(env.pupper.state.final_foot_locations)

            # obs = env.pupper.GetObservation()
            # print(obs[0:3])

            pybullet.stepSimulation()
            last_loop = now

        # print("reset")
        # env.pupper.reset_joint()

    #   env.reset()


if __name__ == '__main__':
    main()
