# 2021 pupper HRL
import os
import inspect
import sys
# Importing the libraries
import os
import pickle
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# Setting the Hyper Parameters
class HighPolicy(object):

    def __init__(self):
        self.nb_steps = 10000
        self.episode_length = 2000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 4
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = "pupper_high_policy"


class LowPolicy(object):

    def __init__(self):
        self.nb_steps = 10000
        self.episode_length = 5000
        self.learning_rate = 0.05
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        # self.env = None
        self.env_name = "pupper_low_policy"

# Multiprocess Exploring the policy on one specific direction and over one episode


_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def ExploreWorker(rank, childPipe, env, args):
    env = env
    nb_inputs = env.observation_space.shape[0]
    normalizer = Normalizer(nb_inputs)
    observation_n = env.reset()
    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:
            observation_n = env.reset()
            childPipe.send(["reset ok"])
            continue
        if message == _EXPLORE:
            # normalizer = payload[0] #use our local normalizer
            policy = payload[1]
            low_p = payload[2]
            direction = payload[3]
            delta = payload[4]
            state = env.reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < low_p.episode_length:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.evaluate(state, delta, direction, low_p)
                state, reward, done, _ = env.step(action)
                # reward = max(min(reward, 1), -1)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards])
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()


# Normalizing the states


class Normalizer(object):

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


# Building the AI


class Policy(object):

    def __init__(self, state_dim, action_dim):
        self.theta = np.zeros((action_dim, state_dim))
        # print("Starting policy theta=", self.theta)

    def evaluate(self, state, delta, direction, hp):
        if direction is None:
            return self.theta.dot(state)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(state)
        else:
            return (self.theta - hp.noise * delta).dot(state)

    def sample_deltas(self, hp):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r, hp):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step


# Exploring the policy on one specific direction and over one episode

class Agent(object):
    def __init__(self,
                 env=None,
                 policy=None,
                 low_policy=None,
                 normalizer=None):
        self.env = env
        self.policy = policy
        self.low_policy = low_policy
        self.normalizer = normalizer
        self.nb_directions = 16
        self.nb_best_directions = 8

    def deploy(self, direction=None, delta=None):
        nb_inputs = self.env.observation_space.shape[0]
        normalizer = Normalizer(nb_inputs)
        state = self.env.reset()
        done = False
        num_plays = 0.
        sum_rewards = 0
        while not done and num_plays < self.low_policy.episode_length:
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction, self.low_policy)
            state, reward, done, _ = self.env.step(action)
            # reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self):
        print("------------------TRAINING------------------------")
        deltas = self.policy.sample_deltas(self.low_policy)
        positive_rewards = [0] * self.nb_directions
        negative_rewards = [0] * self.nb_directions
        print("Deploying Rollouts")
        for i in range(self.nb_directions):
            positive_rewards[i] = self.deploy(direction="positive", delta=deltas[i])

        for i in range(self.nb_directions):
            negative_rewards[i] = self.deploy(direction="negative", delta=deltas[i])
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:self.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        self.policy.update(rollouts, sigma_r, self.low_policy)

        # Printing the final reward of the policy after the update
        reward_evaluation, num_plays = explore(env=self.env,
                                               normalizer=self.normalizer,
                                               policy=self.policy,
                                               direction=None,
                                               delta=None,
                                               low_policy=self.low_policy)
        # print('Step:', step, 'Reward:', reward_evaluation)
        return reward_evaluation, num_plays

    def train_parallel(self, parent_pipes):
        deltas = self.policy.sample_deltas(self.low_policy)
        positive_rewards = [0] * self.nb_directions
        negative_rewards = [0] * self.nb_directions

        if parent_pipes:
            for i in range(self.nb_directions):
                parent_pipe = parent_pipes[i]
                parent_pipe.send([_EXPLORE, [self.normalizer, self.policy, self.low_policy, "positive", deltas[i]]])
            for i in range(self.nb_directions):
                positive_rewards[i] = parent_pipes[i].recv()[0]

            for i in range(self.nb_directions):
                parent_pipe = parent_pipes[i]
                parent_pipe.send([_EXPLORE, [self.normalizer, self.policy, self.low_policy, "negative", deltas[i]]])
            for i in range(self.nb_directions):
                negative_rewards[i] = parent_pipes[i].recv()[0]

        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:self.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        self.policy.update(rollouts, sigma_r, self.low_policy)

        # Printing the final reward of the policy after the update
        reward_evaluation, num_plays = explore(self.env, self.normalizer, self.policy, None, None, self.low_policy)
        # print('Step:', step, 'Reward:', reward_evaluation)
        return reward_evaluation, num_plays

    def save(self, filename):
        with open(filename, 'wb') as filehandle:
            pickle.dump(self.policy.theta, filehandle)

    def load(self, filename):
        with open(filename, 'rb') as filehandle:
            self.policy.theta = pickle.load(filehandle)

    def np_load(self, file_name):
        self.policy.theta = np.load(file_name)


def explore(env, normalizer, policy, direction, delta, low_policy):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < low_policy.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction, low_policy)
        # print(action)
        state, reward, done, _ = env.step(action)
        # reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    obs = env.pupper.GetObservation()
    print(obs)
    return sum_rewards, num_plays


# Training the AI


def train_parallel(env, policy, normalizer, hp, parent_pipes, args):
    for step in range(hp.nb_steps):
        print("------------------TRAINING------------------------")
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas(hp)
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        if parent_pipes:
            for k in range(hp.nb_directions):
                parent_pipe = parent_pipes[k]
                parent_pipe.send([_EXPLORE, [normalizer, policy, hp, "positive", deltas[k]]])
            for k in range(hp.nb_directions):
                positive_rewards[k] = parent_pipes[k].recv()[0]

            for k in range(hp.nb_directions):
                parent_pipe = parent_pipes[k]
                parent_pipe.send([_EXPLORE, [normalizer, policy, hp, "negative", deltas[k]]])
            for k in range(hp.nb_directions):
                negative_rewards[k] = parent_pipes[k].recv()[0]

        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, normalizer, policy, "positive", deltas[k], hp)

            # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, normalizer, policy, "negative", deltas[k], hp)

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        print("Updating policy")
        # Updating our policy
        policy.update(rollouts, sigma_r, hp)

        # Printing the final reward of the policy after the update
        reward_evaluation, num_plays = explore(env=env,
                                               normalizer=normalizer,
                                               policy=policy,
                                               direction=None,
                                               delta=None,
                                               low_policy=hp)
        # print('Step:', step, 'Reward:', reward_evaluation)
        return reward_evaluation, num_plays


# Running the main code


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

"""
if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='Gym environment name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=0)
    parser.add_argument('--movie', help='rgb_array gym movie', type=int, default=0)
    parser.add_argument('--steps', help='Number of steps', type=int, default=10000)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument(
        '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)

    args = parser.parse_args()

    hp = HighPolicy()
    hp.env_name = args.env
    hp.seed = args.seed
    hp.nb_steps = args.steps
    print("seed = ", hp.seed)
    np.random.seed(hp.seed)

    parentPipes = None
    if args.mp:
        num_processes = hp.nb_directions
        processes = []
        childPipes = []
        parentPipes = []

        for pr in range(num_processes):
            parentPipe, childPipe = Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(num_processes):
            p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], hp.env_name, args))
            p.start()
            processes.append(p)

    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    env = gym.make(hp.env_name)
    if args.render:
        env.render(mode="human")
    if args.movie:
        env = wrappers.Monitor(env, monitor_dir, force=True)
    state = env.observation_space.shape[0]
    nb_outputs = env.action_space.shape[0]
    policy = Policy(state, nb_outputs, hp.env_name, args)
    normalizer = Normalizer(state)

    print("start training")
    train(env, policy, normalizer, hp, parentPipes, args)

    if args.mp:
        for parentPipe in parentPipes:
            parentPipe.send([_CLOSE, "pay2"])

        for p in processes:
            p.join()
            """
