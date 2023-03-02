import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy
from utils.layer import ModuleType, mlp_block
from utils.distribution import DiagGaussianDistribution

from .categorical_policy import ActorCriticPolicy as Categorical_ActorCritic
from .categorical_policy import ActorPolicy as Categorical_Actor
from .categorical_policy import PPGActorCritic as Categorical_PPG_ActorCritic

from .gaussian_policy import ActorCriticPolicy as Gaussian_ActorCritic
from .gaussian_policy import ActorPolicy as Gaussian_Actor
from .gaussian_policy import PPGActorCritic as Gaussian_PPG_ActorCritic

from .dqn_policy import DQN_Policy,DuelDQN_Policy,C51DQN_Policy,QRDQN_Policy