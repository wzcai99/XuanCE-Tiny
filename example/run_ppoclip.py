import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormRewardWrapper,NormObservationWrapper,NormActionWrapper,DummyVecEnv
from representation import Identical,MLP,MLP_MT
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import PPOCLIP_Learner
from agent import PPOCLIP_Agent
from environment import MT10_Env
config = get_config("./config/ppoclip/","toy")
envs = [BasicWrapper(MT10_Env(config,i%10)) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)

representation = MLP_MT(space2shape(envs.observation_space),(256,256,),nn.init.orthogonal,nn.LeakyReLU,0)
policy = Gaussian_ActorCritic(envs.action_space,representation,nn.LeakyReLU,nn.init.orthogonal,0)
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
learner = PPOCLIP_Learner(config,policy,optimizer,scheduler,0)
agent = PPOCLIP_Agent(config,envs,policy,learner)
agent.train(config.train_steps)




