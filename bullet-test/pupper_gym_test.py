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
                    -0.1, 0.8, -1.2,
                    0.1, 0.8, -1.2]


def main():
    env = pupper_gym_env.pupperGymEnv(render=True, task=1, height_field=0, hard_reset=False)
    env.reset()
    action = [0.4, 0.2, 0,
              0.4, 0.3, 0,
              0.4, 0.2, 0,
              0.4, 0.1, 0]
    action1 = [0.5, 0, 0.5]
    action2 = [0.5, 0, 0.5]
    action3 = [0.5, 0, 0.5]
    action4 = [0.5, 0, 0.5]
    last_loop = 0
    env.pupper.fr_command.horizontal_velocity = action1[0:2]
    env.pupper.fr_command.height = -0.3
    env.pupper.fl_command.horizontal_velocity = action2[0:2]
    env.pupper.br_command.horizontal_velocity = action3[0:2]
    env.pupper.bl_command.horizontal_velocity = action4[0:2]
    env.pupper.command = [env.pupper.fr_command, env.pupper.fl_command, env.pupper.br_command, env.pupper.bl_command]
    env.pupper.ResetPose()
    # Step the controller forward by dt
    # env.step(action)
    """pybullet.setJointMotorControlArray(
        bodyUniqueId=env.pupper.body_id,
        jointIndices=env.pupper.joint_indices,
        controlMode=pybullet.POSITION_CONTROL,
        targetPositions=list(INIT_MOTOR_ANGLE),
        positionGains=[env.pupper.motor_kp] * 12,
        velocityGains=[env.pupper.motor_kv] * 12,
        forces=[env.pupper.motor_max_torque] * 12,
    )
    for _ in range(10):
        env.pupper.ResetPose()"""
    while True:
        now = time.time()
        if now - last_loop >= 0.01:
            # Check if we should transition to "deactivated"
            # env.pupper.ResetPose()
            # Step the controller forward by dt
            env.step(action)
            # env.pupper.TG.run(state=env.pupper.state, command=env.pupper.command)
            # env.pupper.foot_position2motor_angle(env.pupper.state.final_foot_locations)
            obs = env.pupper.GetObservation()
            print(obs[0:3])

            pybullet.stepSimulation()
            last_loop = now

        # print("reset")
        # env.pupper.reset_joint()

    #   env.reset()


if __name__ == '__main__':
    main()
