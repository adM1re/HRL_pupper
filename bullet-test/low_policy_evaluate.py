import numpy as np
import os
import sys
# sys.path.append("./")
import argparse
import pupper
import pupper_gym_env
import pybullet_data
import pybullet
import multiprocessing as mp
from multiprocessing import Pipe
from pupper_gym_env import pupperGymEnv
from ars_lib.ars import *
# Messages for Pipe
_RESET = 1
_CLOSE = 2
_EXPLORE = 3
result_file_name = "low_policy_evaluate_result"
result_path = "result"
model_path = "model"
model_file_name = "low_policy_trained"


def main():
    mp.freeze_support()

    describe = "Pupper ARS Agent Policy "
    # aim to training pupper walking
    parse = argparse.ArgumentParser(description=describe)
    parse.add_argument("-t", "--task", type=int, default=1, help="Task Number")
    parse.add_argument("-s", "--seed", help="Random Seed", type=int, default=0)
    parse.add_argument("-r", "--render", help="Is Rendering", type=bool, default=0)
    parse.add_argument("-m", "--mp", help="Enable Multiprocessing", type=bool, default=0)
    parse.add_argument("-p", "--policy", type=str, default="")
    parse.add_argument("-a", "--agent", type=int, default=0)
    args = parse.parse_args()
    seed = 13
    seed = args.seed
    print("Seed:{}".format(seed))
    max_time_steps = 4e6
    eval_freq = 1
    save_model = True
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    task_no = args.task

    env = pupperGymEnv(render=True,
                       task=task_no,
                       height_field=0)
    env.seed(seed)
    np.random.seed(seed)
    state_dim = env.observation_space.shape[0]
    print("STATE DIM:{}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM:{}".format(action_dim))
    # max_action = float(env.action_space.high)

    low_policy = LowPolicy()
    # high_policy = HighPolicy()
    # low_policy.env = env

    env.reset()
    episode_reward = 0
    episode_time_steps = 0
    episode_num = args.agent
    normalizer = Normalizer(state_dim)
    policy = Policy(state_dim, action_dim)
    low_policy_agent = Agent(env, policy, low_policy, normalizer)
    agent_num = args.agent
    if os.path.exists(model_path + "/" + str(seed) + "/" + model_file_name + "agent" + str(agent_num) + ".npy"):
        print("Loading Existing agent:")
        print(model_path + "/" + str(seed) + "/" + model_file_name + "agent" + str(agent_num) + ".npy")
        low_policy_agent.np_load(model_path + "/" + str(seed) + "/" + model_file_name + "agent" + str(agent_num) + ".npy")
        print("Starting policy theta=", low_policy_agent.policy.theta)
    else:
        print("Start New Training")
        print("Starting policy theta=", low_policy_agent.policy.theta)

    print("Started Pupper Training Env")
    t = 0
    try:
        result = np.load(result_path + "/low_policy_resultseed" + str(seed) + ".npy")
        er = 0
    except Exception as e:
        er = 1
        print("No trained result")
    while t < (int(max_time_steps)):
        episode_reward, episode_time_steps = explore(env=env, policy=policy, normalizer=normalizer, direction=None,
                                                     delta=None, low_policy=low_policy)
        t += episode_time_steps
        print(
            "Total T: {} Episode Num: {} Episode T: {} Reward: {:.2f} REWARD PER STEP: {:.2f}".format(
                t + 1,
                episode_num,
                episode_time_steps,
                episode_reward,
                episode_reward / float(episode_time_steps)
            )
        )
        if episode_num == 0:
            if er == 1:
                result = np.array(
                    [[episode_reward, episode_reward / float(episode_time_steps)]]
                )
            else:
                new_result = np.array(
                    [[episode_reward, episode_reward / float(episode_time_steps)]]
                )
                result = np.concatenate((result, new_result))
        else:
            new_result = np.array(
                [[episode_reward, episode_reward / float(episode_time_steps)]]
            )
            result = np.concatenate((result, new_result))
        # Save training result
        np.save(result_path + "/" + str(result_file_name) + "seed" + str(seed) + ".npy", result)
        # Save training model
        episode_num += 1


if __name__ == '__main__':
    main()
