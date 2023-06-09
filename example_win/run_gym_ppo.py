import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
# import envpool
import numpy as np
import random
from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,DummyVecEnv,RewardNorm,ObservationNorm,ActionNorm
# from xuance.environment import EnvPool_Wrapper,EnvPool_ActionNorm,EnvPool_RewardNorm,EnvPool_ObservationNorm
from xuance.representation import MLP
from xuance.policy import Categorical_ActorCritic,Gaussian_ActorCritic
from xuance.learner import PPO_Learner
from xuance.agent import PPO_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="config/ppo/")
    parser.add_argument("--domain",type=str,default="mujoco") # default: same config.yaml for env from the same domain
    parser.add_argument("--env_id",type=str,default="BipedalWalker-v3") # LunarLander-v2, BipedalWalker-v3, ...
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=True)
    args = parser.parse_known_args()[0]
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config,args.domain)
    set_seed(config.seed)
   
    # in some cases, the training environment is different with the testing environment
    def build_train_envs(): 
        env = gym.make(args.env_id, render_mode="rgb_array")
        envs = [BasicWrapper(env) for _ in range(config.nenvs)]
        return ObservationNorm(config, RewardNorm(config, ActionNorm(DummyVecEnv(envs)), train=True), train=True)

    def build_test_envs(): 
        env = gym.make(args.env_id, render_mode="rgb_array")
        envs = [BasicWrapper(env) for _ in range(2)]
        return ObservationNorm(config, RewardNorm(config, ActionNorm(DummyVecEnv(envs)), train=False), train=False)

    train_envs = build_train_envs()

    representation = MLP(space2shape(train_envs.observation_space),(256,256),nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = Gaussian_ActorCritic(train_envs.action_space,representation,nn.init.orthogonal_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
    learner = PPO_Learner(config,policy,optimizer,scheduler,device)
    agent = PPO_Agent(config,train_envs,policy,learner)
    
    agent.benchmark(build_test_envs(),config.train_steps,config.evaluate_steps,render=args.render)