import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormRewardWrapper,NormObservationWrapper,NormActionWrapper,DummyVecEnv
from representation import Identical,MLP
from policy import DuelDQN_Policy
from learner import DQN_Learner
from agent import DQN_Agent

config = get_config("./config/dqn/","toy")
envs = [BasicWrapper(gym.make("CartPole-v1")) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
representation = MLP(space2shape(envs.observation_space),(128,),nn.init.orthogonal,nn.LeakyReLU,0)
policy = DuelDQN_Policy(envs.action_space,representation,nn.init.orthogonal,0)
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate,eps=1e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
learner = DQN_Learner(config,policy,optimizer,scheduler,0)
agent = DQN_Agent(config,envs,policy,learner)
#agent.train(config.train_steps)
#agent.test("model-Mon Nov 28 17:20:26 2022-10000.pth",10000,0.0)



