import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormRewardWrapper,NormObservationWrapper,NormActionWrapper,DummyVecEnv
from representation import Identical,MLP
from policy import TD3Policy
from learner import TD3_Learner
from agent import TD3_Agent

config = get_config("./config/td3/","mujoco")
envs = [NormActionWrapper(BasicWrapper(gym.make("HalfCheetah-v4"))) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
representation = MLP(space2shape(envs.observation_space),(256,),nn.LeakyReLU,nn.init.xavier_uniform,"cuda:0")
policy = TD3Policy(envs.action_space,representation,nn.init.xavier_uniform,"cuda:0")
actor_optimizer = torch.optim.Adam(policy.actor_parameters,config.actor_lr_rate,eps=1e-5)
critic_optimizer = torch.optim.Adam(policy.critic_parameters,config.critic_lr_rate,eps=1e-5)
actor_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
critic_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
learner = TD3_Learner(config,policy,[actor_optimizer,critic_optimizer],[actor_scheduler,critic_scheduler],"cuda:0")
agent = TD3_Agent(config,envs,policy,learner)
agent.train(config.train_steps)
#agent.test("model-Mon Nov 28 17:20:26 2022-10000.pth",10000,0.0)



