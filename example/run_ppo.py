import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import numpy as np
import envpool
from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,DummyVecEnv,RewardNorm,ObservationNorm,ActionNorm
from xuance.environment import EnvPool_Wrapper,EnvPool_ActionNorm,EnvPool_RewardNorm,EnvPool_ObservationNorm
from xuance.representation import MLP
from xuance.policy import Categorical_ActorCritic,Gaussian_ActorCritic
from xuance.learner import PPO_Learner
from xuance.agent import PPO_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/ppo/")
    parser.add_argument("--domain",type=str,default="mujoco")
    parser.add_argument("--env_id",type=str,default="InvertedPendulum-v4")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config,args.domain)
    
    # define a vector environment
    # train_envs = [BasicWrapper(gym.make(args.env_id,render_mode="rgb_array")) for i in range(config.nenvs)]
    # train_envs = DummyVecEnv(train_envs)
    # train_envs = ActionNorm(train_envs)
    # train_envs = RewardNorm(config,train_envs,train=(args.pretrain_weight is None))
    # train_envs = ObservationNorm(config,train_envs,train=(args.pretrain_weight is None))
    
    # define a envpool environment
    train_envs = envpool.make(args.env_id,"gym",num_envs=config.nenvs)
    train_envs = EnvPool_Wrapper(train_envs)
    train_envs = EnvPool_ActionNorm(train_envs)
    train_envs = EnvPool_RewardNorm(config,train_envs)
    train_envs = EnvPool_ObservationNorm(config,train_envs)
    representation = MLP(space2shape(train_envs.observation_space),(256,256),nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = Gaussian_ActorCritic(train_envs.action_space,representation,nn.init.orthogonal_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
    learner = PPO_Learner(config,policy,optimizer,scheduler,device)
    agent = PPO_Agent(config,train_envs,policy,learner)
    
    # in many cases, the training environment is different with the testing environment
    def build_test_env():
        test_envs = [BasicWrapper(gym.make(args.env_id,render_mode="rgb_array")) for _ in range(1)]
        test_envs = DummyVecEnv(test_envs)
        test_envs = ActionNorm(test_envs)
        test_envs = RewardNorm(config,test_envs,train=False)
        test_envs = ObservationNorm(config,test_envs,train=False)
        return test_envs
    agent.benchmark(build_test_env,config.train_steps,config.evaluate_steps,render=args.render)
    
   