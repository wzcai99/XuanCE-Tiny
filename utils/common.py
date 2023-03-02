import gym
import os
import yaml
import scipy.signal
import numpy as np
from argparse import Namespace

def get_config(dir_name, args_name):
    with open(os.path.join(dir_name, args_name + ".yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, args_name + ".yaml error: {}".format(exc)
    return Namespace(**config_dict)


def create_directory(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"

def space2shape(observation_space: gym.Space):
    if isinstance(observation_space, gym.spaces.Dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    else:
        return observation_space.shape
    
def discount_cumsum(x, discount=0.99):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
