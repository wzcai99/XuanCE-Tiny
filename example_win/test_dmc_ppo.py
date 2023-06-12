import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
# import gym
# import envpool
import xuance.environment.custom_envs.dmc as dmc
import numpy as np
import random
from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,DummyVecEnv,RewardNorm,ObservationNorm,ActionNorm
from xuance.representation import MLP
from xuance.policy import Gaussian_ActorCritic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    
    parser.add_argument("--config",type=str,default="config/ppo/")
    parser.add_argument("--domain",type=str,default="walkerStand") # default: same config.yaml for env from the same domain
    parser.add_argument("--task_id",type=str,default="walker") # walker, swimmer, ...
    parser.add_argument("--env_id",type=str,default="stand") # stand, walk, ...
    parser.add_argument("--time_limit",type=int,default=150)
    
    parser.add_argument("--pretrain_weight",type=str,default=r"D:\zzm_codes\xuance_TneitapSimHand\models\walkerStand0612-2\ppo-79811\best_model.pth")

    parser.add_argument("--render",type=bool,default=True)

    args = parser.parse_known_args()[0]
    return args

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

if __name__ == "__main__":

    args = get_args()
    device = args.device
    config = get_config(args.config,args.domain)

    def build_test_envs(): 
        env = dmc.DMControl(args.task_id,args.env_id, args.time_limit)
        envs = [BasicWrapper(env) for _ in range(8)]
        envs = RewardNorm(config, ActionNorm(DummyVecEnv(envs)), train=False)
        envs.load_rms() # os.path.join(self.save_dir,"reward_stat.npy")
        envs = ObservationNorm(config, envs, train=False)
        envs.load_rms()
        return envs
    
    envs = build_test_envs()

    mlp_hiddens = tuple(map(int, config.mlp_hiddens.split(",")))
    representation = MLP(space2shape(envs.observation_space),mlp_hiddens,nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal_,device)
    policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    print("use weights: ", args.pretrain_weight)
    policy.eval()

    obs,infos = envs.reset() # (nenvs, 24)

    test_episode = 1000
    current_episode = 0
    while current_episode < test_episode:
        print("[%03d]"%(current_episode))
        envs.render("human")
        # obs_Tsor = torch.from_numpy(obs['observation']).float().to(policy.actor.device)
        _,act_Distrib,_ = policy.forward(obs) # (nenvs, 6)
        act_Tsor = act_Distrib.sample()
        next_obs,rewards,terminals,trunctions,infos = envs.step(to_cpu(act_Tsor))
        for i in range(envs.num_envs):
            if terminals[i] == True or trunctions[i] == True: 
                current_episode += 1
        obs = next_obs