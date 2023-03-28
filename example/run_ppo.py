import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv,RewardNorm,ObservationNorm,DMControl
from representation import MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import PPO_Learner
from agent import PPO_Agent

# define hyper-parameters
device = "cuda:0"
config = get_config("./config/ppo/","mujoco")
# define the vector environment
#envs = [NormActionWrapper(BasicWrapper(DMControl("pendulum","swingup",200))) for i in range(config.nenvs)]
envs = [NormActionWrapper(BasicWrapper(gym.make("HalfCheetah-v4",render_mode="rgb_array"))) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
envs = ObservationNorm(RewardNorm(envs))
# network and training
representation = MLP(space2shape(envs.observation_space),(128,128),nn.LeakyReLU,nn.init.orthogonal,device)
policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal,device)
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
learner = PPO_Learner(config,policy,optimizer,scheduler,device)
agent = PPO_Agent(config,envs,policy,learner)
agent.benchmark(config.train_steps,config.evaluate_steps)




