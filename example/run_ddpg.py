import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,ActionNorm,DummyVecEnv
from xuance.representation import MLP
from xuance.policy import DDPGPolicy
from xuance.learner import DDPG_Learner
from xuance.agent import DDPG_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/ddpg/")
    parser.add_argument("--domain",type=str,default="mujoco")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config,args.domain)
    envs = [BasicWrapper(gym.make("Humanoid-v4",render_mode='rgb_array')) for i in range(config.nenvs)]
    envs = DummyVecEnv(envs)
    envs = ActionNorm(envs)
    representation = MLP(space2shape(envs.observation_space),(256,),nn.LeakyReLU,nn.init.xavier_uniform_,device)
    policy = DDPGPolicy(envs.action_space,representation,nn.init.xavier_uniform_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    actor_optimizer = torch.optim.Adam(policy.actor_parameters,config.actor_lr_rate,eps=1e-5)
    critic_optimizer = torch.optim.Adam(policy.critic_parameters,config.critic_lr_rate,eps=1e-5)
    actor_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    critic_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    learner = DDPG_Learner(config,policy,[actor_optimizer,critic_optimizer],[actor_scheduler,critic_scheduler],device)
    agent = DDPG_Agent(config,envs,policy,learner)
    agent.benchmark(config.train_steps,config.evaluate_steps,render=args.render)



