import pybullet
import time
import pybullet_data
import gym
import numpy as np
import pupper_gym_test as pgt


# 初始化环境
# env.reset()
# 循环1000次
def main():
    """ The main Function"""
    print("                                       Start Pupper Gym Env Test")
    env = pgt.pupperGymEnvTest(action_repeat=1, render=True)
    env.reset()
    action = []
    while True:
        env.step(action)  # 与环境交互


if __name__ == '__main__':
    main()
