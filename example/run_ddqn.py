import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import argparse
from utils.common import space2shape,get_config
from environment import BasicWrapper,DummyVecEnv
from representation import MLP
from policy import DQN_Policy
from learner import DDQN_Learner
from agent import DQN_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/dqn/")
    parser.add_argument("--domain",type=str,default="toy")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config,args.domain)
    envs = [BasicWrapper(gym.make("LunarLander-v2",render_mode='rgb_array')) for i in range(config.nenvs)]
    envs = DummyVecEnv(envs)
    representation = MLP(space2shape(envs.observation_space),(256,),nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = DQN_Policy(envs.action_space,representation,nn.init.orthogonal_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate,eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    learner = DDQN_Learner(config,policy,optimizer,scheduler,device)
    agent = DQN_Agent(config,envs,policy,learner)
    agent.benchmark(config.train_steps,config.evaluate_steps,render=args.render)




