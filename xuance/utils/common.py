import gym
import os
import time 
import yaml
import scipy.signal
import numpy as np
from terminaltables import AsciiTable
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
            os.makedirs(current_dir, exist_ok=True)
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


################# Time Func For Log ##############################
def get_time_hm()->str:
    localtime = time.localtime(time.time())
    return "%02d:%02d"%(localtime.tm_hour, localtime.tm_min)

def get_time_full()->str:
    return time.asctime().replace(":", "_")#.replace(" ", "_")


###################################################################


def log_the_table(title, info_table: list, txt_writer):
    def log_the_str(*print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = txt_writer)
        print("")
        print("", file = txt_writer)
        return
    
    table = AsciiTable(info_table, title).table
    log_the_str(table)

def summarize_ppo_config(pConfig):
        
        info_table = [['item', 'detail']]
        info_table.append(["# interact envs", pConfig.nenvs])
        info_table.append(["# total train_steps", str(pConfig.train_steps/1000) + " K"])
        info_table.append(["# train-test epochs", pConfig.train_steps // pConfig.evaluate_steps])
        info_table.append([" ", " "])
        info_table.append(["# train_step per train", pConfig.evaluate_steps])
        info_table.append(["# policy max updates per train_step", pConfig.evaluate_steps//pConfig.nsize])
        info_table.append(["# updates to save model", str(pConfig.save_model_frequency/1000) + " K"])
        info_table.append(["# data samples per train_step", pConfig.nminibatch * pConfig.nepoch])
        info_table.append(["# data batchsize per update", pConfig.nenvs * pConfig.nsize // pConfig.nsize])
        info_table.append([" ", " "])
        info_table.append(["target_kl for erly-stopping train_step", pConfig.target_kl])
        info_table.append(["base lr for Adam", pConfig.lr_rate])
        info_table.append([" ", " "])
        info_table.append(["loss-graph log timestep <=", str((pConfig.train_steps//pConfig.nsize)*pConfig.nminibatch * pConfig.nepoch/1000)+" K"])
        info_table.append(["reward-graph log timestep ==", str(pConfig.train_steps * pConfig.nenvs/1000000)+" M"])
        os.makedirs(os.path.join(pConfig.logdir, pConfig.env_name), exist_ok=True)
        logfile_path = os.path.join(pConfig.logdir, pConfig.env_name, "config_settings.txt")
        txt_writer = open(logfile_path, 'a+')
        log_the_table("config" , info_table, txt_writer)
        txt_writer.close()