import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,DummyVecEnv,RewardNorm,ObservationNorm
from representation import MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import PPOCLIP_Learner
from agent import PPO_Agent

# define hyper-parameters
device = "cuda:0"
config = get_config("./config/ppoclip/","mujoco")
# define the vector environment
envs = [BasicWrapper(gym.make("Ant-v4")) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
envs = ObservationNorm(RewardNorm(envs))
# network and training
representation = MLP(space2shape(envs.observation_space),(256,256),nn.LeakyReLU,nn.init.orthogonal,device)
policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal,device)
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
learner = PPOCLIP_Learner(config,policy,optimizer,scheduler,device)
agent = PPO_Agent(config,envs,policy,learner)
agent.train(config.train_steps)




