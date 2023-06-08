import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import envpool
import numpy as np
import random
from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,ActionNorm,DummyVecEnv
from xuance.environment import EnvPool_Wrapper,EnvPool_RewardNorm,EnvPool_ActionNorm,EnvPool_ObservationNorm
from xuance.representation import MLP
from xuance.policy import TD3Policy
from xuance.learner import TD3_Learner
from xuance.agent import TD3_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/td3/")
    parser.add_argument("--domain",type=str,default="mujoco")
    parser.add_argument("--env_id",type=str,default="HalfCheetah-v4")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
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
   
    # define a envpool environment
    train_envs = envpool.make(args.env_id,"gym",num_envs=config.nenvs)
    train_envs = EnvPool_Wrapper(train_envs)
    train_envs = EnvPool_ActionNorm(train_envs)
    
    representation = MLP(space2shape(train_envs.observation_space),(256,),nn.LeakyReLU,nn.init.xavier_uniform_,device)
    policy = TD3Policy(train_envs.action_space,representation,nn.init.xavier_uniform_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    actor_optimizer = torch.optim.Adam(policy.actor_parameters,config.actor_lr_rate,eps=1e-5)
    critic_optimizer = torch.optim.Adam(policy.critic_parameters,config.critic_lr_rate,eps=1e-5)
    actor_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    critic_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    learner = TD3_Learner(config,policy,[actor_optimizer,critic_optimizer],[actor_scheduler,critic_scheduler],device)
    agent = TD3_Agent(config,train_envs,policy,learner)
    
    # in many cases, the training environment is different with the testing environment
    def build_test_env():
        test_envs = [BasicWrapper(gym.make(args.env_id,render_mode='rgb_array')) for _ in range(1)]
        test_envs = DummyVecEnv(test_envs)
        test_envs = ActionNorm(test_envs)
        return test_envs
    test_envs = build_test_env()
    agent.benchmark(test_envs,config.train_steps,config.evaluate_steps,render=args.render)



