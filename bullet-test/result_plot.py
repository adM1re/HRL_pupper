import numpy as np
import argparse
# import matplot
result_path = "result/"
evaluate_name = "low_policy_evaluate_result"
training_name = "low_policy_result"

describe = "Pupper ARS Low Policy Result"
parse = argparse.ArgumentParser(description=describe)
parse.add_argument("-t", "--task", type=int, default=1, help="Task Number")
parse.add_argument("-s", "--seed", help="Random Seed", type=int, default=0)
parse.add_argument("-r", "--render", help="Is Rendering", type=bool, default=0)
parse.add_argument("-m", "--mp", help="Enable Multiprocessing", type=bool, default=0)
parse.add_argument("-p", "--policy", type=str, default="")
parse.add_argument("-a", "--agent", type=int, default=2)
args = parse.parse_args()
seed = args.seed
seed = 3
result = np.load(result_path + training_name + "seed" + str(seed) + ".npy")
# print(result)
agent_num = []
per_step_reward = []
for i in range(len(result)):
    agent_num.append(i)
    per_step_reward.append(result[i, 1])
print(agent_num)
print(per_step_reward)
# matplot(agent_num, per_step_reward)
