import pickle
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
result_file_name = "low_policy_result"
result_path = "result"
model_path = "model"
model_file_name = "low_policy_trained"


def main():
    seed = 0
    result = np.load(result_path + "/" + str(result_file_name) + "seed" + str(seed) + ".npy")
    print(result)


if __name__ == '__main__':
    main()



