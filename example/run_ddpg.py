import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv
from representation import MLP
from policy import DDPGPolicy
from learner import DDPG_Learner
from agent import DDPG_Agent

config = get_config("./config/ddpg/","mujoco")
envs = [NormActionWrapper(BasicWrapper(gym.make("HalfCheetah-v4"))) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
representation = MLP(space2shape(envs.observation_space),(256,),nn.LeakyReLU,nn.init.xavier_uniform,"cuda:0")
policy = DDPGPolicy(envs.action_space,representation,nn.init.xavier_uniform,"cuda:0")
actor_optimizer = torch.optim.Adam(policy.actor_parameters,config.actor_lr_rate,eps=1e-5)
critic_optimizer = torch.optim.Adam(policy.critic_parameters,config.critic_lr_rate,eps=1e-5)
actor_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
critic_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
learner = DDPG_Learner(config,policy,[actor_optimizer,critic_optimizer],[actor_scheduler,critic_scheduler],"cuda:0")
agent = DDPG_Agent(config,envs,policy,learner)
agent.benchmark(config.train_steps,config.evaluate_steps)



