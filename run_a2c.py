from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormRewardWrapper,NormObservationWrapper,NormActionWrapper,DummyVecEnv
from representation import Identical,MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import A2C_Learner
from agent import A2C_Agent

config = get_config("./config/a2c/","toy")
envs = [NormRewardWrapper(BasicWrapper(gym.make("CartPole-v1"))) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
representation = MLP(space2shape(envs.observation_space),(128,128,),nn.init.orthogonal,nn.LeakyReLU,0)
policy = Categorical_ActorCritic(envs.action_space,representation,nn.Tanh,nn.init.orthogonal,0)
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
learner = A2C_Learner(config,policy,optimizer,scheduler,0)
agent = A2C_Agent(config,envs,policy,learner)
agent.train(config.train_steps)



