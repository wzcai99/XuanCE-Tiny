import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import envpool

from xuance.utils.common import space2shape,get_config
from xuance.environment import BasicWrapper,DummyVecEnv,Atari
from xuance.environment import EnvPool_Wrapper,EnvPool_RewardNorm,EnvPool_ActionNorm,EnvPool_ObservationNorm
from xuance.representation import CNN
from xuance.policy import DQN_Policy
from xuance.learner import DDQN_Learner
from xuance.agent import DQN_Agent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/ddqn/")
    parser.add_argument("--domain",type=str,default="atari")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    # define hyper-parameters
    device = args.device
    config = get_config(args.config,args.domain)
    #define a vector enviroment
    # train_envs = [BasicWrapper(Atari("PongNoFrameskip-v4",render_mode="rgb_array")) for i in range(config.nenvs)]
    # train_envs = DummyVecEnv(train_envs)
    
    #define a envpool environment
    train_envs = envpool.make("Pong-v5","gym",num_envs=config.nenvs)
    train_envs = EnvPool_Wrapper(train_envs)
    
    #envs = EnvPool_Wrapper(envpool.make("Pong-v5","gym",num_envs=config.nenvs))
    representation = CNN(space2shape(train_envs.observation_space),(16,16,32,32),(8,6,4,4),(2,2,2,2),nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = DQN_Policy(train_envs.action_space,representation,nn.init.orthogonal_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate,eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.training_frequency)
    learner = DDQN_Learner(config,policy,optimizer,scheduler,device)
    agent = DQN_Agent(config,train_envs,policy,learner)
    
    def build_env_fn():
        test_envs = [BasicWrapper(Atari("PongNoFrameskip-v4",render_mode="rgb_array")) for _ in range(1)]
        test_envs = DummyVecEnv(test_envs)
        return test_envs
    agent.benchmark(build_env_fn,config.train_steps,config.evaluate_steps,0,render=args.render)




