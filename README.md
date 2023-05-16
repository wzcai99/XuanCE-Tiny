## XuanCE: A simple framework and implementations of reinforcement learning algorithms ##
XuanCE is a reinforcement learning algorithm platform which supports multiple deep learning frameworks (Pytorch, TensorFlow, Mindspore) and both multi-agent RL and single-agent RL methods.
This repository is a pruned version of the original project [XuanPolicy](https://openi.pcl.ac.cn/OpenRelearnware/XuanPolicy) with only Pytorch-based implementations and single-agent RL algorithms. 
We make this repo as much as highly-modularized and clean to be friendly for the RL starters.
And the code is also compatiable and easy-to-use for researchers to implement their own RL algorithms or verify their ideas.

- For example, if you want to benchmark the RL algorithms on some novel problems, just following the example provided in <strong><em> xuance/environment/custom_envs/dmc.py </em></strong> to formalize a novel problem into Markov Decision Process (MDP) into a gym-based wrapper. A tutorial is provided [here](https://xuance.readthedocs.io/en/latest/).
- If you want to try some advanced representation network, just following the example provided in <strong><em> xuance/representation/network.py </em></strong> to define a new class based on <strong><em> torch.nn.Module</em></strong>. A tutorial is provided [here](https://xuance.readthedocs.io/en/latest/).
- If you figure out a better way for RL optimization process, just add a learner similar to the <strong><em> xuance/learner/xxx.py </em></strong> and define your own loss function. You can compare difference in <strong><em> xuance/learner/a2c.py </em></strong> and <strong><em> xuance/learner/ppo.py </em></strong> for your own implementation. 
- If you propose a more efficient memory buffer and experience replay scheme, just add your own memory buffer class in <strong><em> xuance/utils/memory.py </em></strong> and replace the memory used in the <strong><em> xuance/agents/xxx.py </em></strong>
- ...
- More details of the usage can be found in the [documentions](https://xuance.readthedocs.io/en/latest/).

In summary, our high-modularized design allows us to focus on unit design and improvements with other parts untouched.

Currently, this repo supports the following RL algorithms which are:
- Advantage Actor-Critic(A2C)
- Proximal Policy Optimization(PPO)
- Deep Deterministic Policy Gradient(DDPG)
- Twin Delayed DDPG(TD3)
- Deep-Q Network(DQN)
- Dueling-Q Network(Duel-DQN)
- Double-Q Network(DDQN)

## Installation ##
You can clone this repository and install an editable version locally:
```
$ conda create -n xuance python=3.9
$ conda activate xuance
$ git clone https://github.com/wzcai99/XuanCE.git
$ cd XuanCE
$ pip install -e .
```

## Quick Start ## 
You can run the RL algorithms with the provided examples,
```
$ python -m example.run_a2c
```
or follow the below step-by-step instructions. 
Here we show how to integrate all the components and run the advantage-actor-critic (A2C) method. 

First, define a configuration file contains all the hyper-parameters in the format of PyYAML.
```
nenvs: 16  # the environments running in parallel
nsize: 64  # interactions for data-collection per env for optimization
nminibatch: 1 # batchsize for iteration = nenvs * nsize // nminibatch
nepoch: 1   # policy iteration times every nenvs*nsize = nminibatch * nepoch

vf_coef: 0.25  # the weight for critic loss
ent_coef: 0.00 # the weight for entropy regularization
clipgrad_norm: 0.5 # gradient-clip norm
lr_rate: 0.0005 # learning rate

save_model_frequency: 100 # every 100 policy updates, save a model
train_steps: 187500   # total interaction steps = train_steps * nenvs
evaluate_steps: 10000 # every 10000 iteractions, evaluate the policy

gamma: 0.99 # discount-factor
tdlam: 0.95 # gae-lambda

logdir: "./logs/HalfCheetah/a2c/" #directory to save the logging file for tensorboard
modeldir: "./models/HalfCheetah/a2c/" #directory to save the policy model
```
Import some relavant packages:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
from xuance.utils.common import space2shape,get_config
```
Parse some arguments: 
```
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--config",type=str,default="./config/a2c/")
    parser.add_argument("--domain",type=str,default="mujoco")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args
args = get_args()
device = args.device
config = get_config(args.config,args.domain)
```
Note that the argument <strong><em>config</em></strong> is the directory saving the PyYAML file and the argument <strong><em>domain</em></strong> is the filename of the PyYAML file.

Define a vector of environments:
```
from xuance.environment import BasicWrapper,DummyVecEnv,RewardNorm,ObservationNorm,ActionNorm
envs = [BasicWrapper(gym.make("HalfCheetah-v4",render_mode='rgb_array')) for i in range(config.nenvs)]
envs = DummyVecEnv(envs)
# To include action normalization,
envs = ActionNorm(envs)
# To include observation normalization,
envs = ObservationNorm(config,envs,train=args.pretrain_weight is None)
# To include reward normalization,
envs = RewardNorm(config,envs,train=args.pretrain_weight is None)
```
Define a representation network:
```
from xuance.representation import MLP
representation = MLP(space2shape(envs.observation_space),(128,128),nn.LeakyReLU,nn.init.orthogonal_,device)
```
Define the policy:
```
from xuance.policy import Categorical_ActorCritic,Gaussian_ActorCritic
# For Discrete Action Space:
policy = Categorical_ActorCritic(envs.action_space,representation,nn.init.orthogonal_,device)
# For Continuous Action Space:
policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal_,device)
```
If you want to load a pre-trained policy weight:
```
if args.pretrain_weight:
    policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
```
Define an optimizer and a learning rate scheduler:
```
optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
```
Define the RL learner and RL agent:
```
from xuance.learner import A2C_Learner
from xuance.agent import A2C_Agent
learner = A2C_Learner(config,policy,optimizer,scheduler,device)
agent = A2C_Agent(config,envs,policy,learner)
```
Train the RL agent or test the RL agent:
```
agent.train(config.train_steps)
agent.test(20,args.render) # test for 20 episodes
```
You can also run a benchmark experiment:
```
agent.benchmark(config.train_steps,config.evaluate_steps,render=args.render)
```

After that, you can use tensorboard or the plotter to see the training curve.
```
$ tensorboard --logdir=./logs/ --port=6007
```
```
$ python -m xuance.utils.tensorboard_plotter --env_name=HalfCheetah --log_dir=./logs/ --y_smooth=0.9 --x_smooth=1000
```
<div align=center>
<figure>
<img src="./figures/plotter.png" width="45%" height=225/>
<img src="./figures/tensorboard.png" width="45%" height=225/>
</figure>
</div>


<!-- <img decoding="async" src="./figures/plotter.png" width="40%" height=250>
<img decoding="async" src="./figures/tensorboard.png" width="40%" height=250> -->

## Benchmark Results (ToDo) ##
We will provide a complete benchmark results in MuJoCo and Atari games. The performance are evaluated with the best model during the training and we report the average scores with 3 different random seeds.

|  Environment (3M)  | xuance-a2c | tianshou-a2c | xuance-ppo | tianshou-ppo |
|  :----:  | :----:  |  :----:  | :----:  | :----:  |
| HalfCheetah  | <strong>2860.9</strong> | 2377.3  | 5749.3 | <strong>7337</strong> |
| Ant  | 1943.9 | <strong>5236.8</strong>  | 3859.9 | <strong>4079.3</strong> |
| Walker2d  | <strong>2066.6</strong> | 1805.4  | 4339.3 | <strong>4895.6</strong> |
| Hopper  | <strong>2531.1</strong> | 1608.6  | <strong>3520.2</strong> | 3127.7 |
| Swimmer  | <strong>46.9</strong> | 40.2  | <strong>108.9</strong> | 81.4 |
| Reacher  | -13.5 | <strong>-5.2</strong>  | -4.88 | <strong>-3.7</strong> |
| Humanoid  | 392.4 | <strong>5316.6</strong>  | 1057.3 | <strong>1369.7</strong> |
| InvertedPendulum  | 1000.0 | 1000.0  | 1000.0 | 1000.0 |
| InvertedDoublePendulum | <strong>9346.6</strong> | 9351.3  | <strong>9468.2</strong> | 9231.3 |

|  Environment (1M)  | xuance-a2c | published-a2c | xuance-ppo | published-ppo |
|  :----:  | :----:  |  :----:  | :----:  | :----:  |
| HalfCheetah  | <strong>1264.35</strong> | 1000  | <strong>4628.4</strong> | 1800 |
| Ant  |  | /  | 2810.7 | / |
| Walker2d  |  | 850  | <strong>4318.6</strong> | 3460 |
| Hopper  |  | 900  | <strong>3450.1</strong> | 2330 |
| Swimmer |  | 31 | <strong>108.9</strong> | 108 |
| Reacher |  | -24  | -8.1 | <strong>-7</strong> |
| Humanoid  |  | /  | 705.5 | / |
| InvertedPendulum  |  | 1000  | 1000.0 | 1000 |
| InvertedDoublePendulum  |  | 7100  | <strong>9359.1</strong> | 8000 |

|  Environment (1M)  | xuance-ddpg | published-ddpg | tianshou-ddpg | xuance-td3 | published-td3 | tianshou-td3 | 
|  :----:  | :----:  |  :----:  | :----:  | :----:  | :----:  | :----:  |
|  HalfCheetah   | 10093 |  8577.3  | <strong>11718</strong>  |   | 9637.0  | 10201.2  |
|  Ant           | <strong>1472.8</strong>  |  888.8  | 990.4  |   | 4372.0  | 5116.4  |
|  Walker2d      | 2443.7  |  <strong>3098.1</strong>  | 1400.6  |   | 4682.8  | 3982.4  |
|  Hopper        | <strong>3434.9</strong>  |  1860.0  | 2197.0  |   | 3564.1  | 3472.2  |
|  Swimmer        | 67.7  |  /  | <strong>144.1</strong>  |   | /  | 104.2  |
|  Reacher       | -4.05  |  -4.01  | <strong>-3.3</strong>  |   | -3.6  | -2.7  |
|  Humanoid      |   |  /  | 177.3  |   | /  | 5189.5  |
|  InvertedPendulum  | 1000.0  |  1000.0  | 1000.0  |   | 1000.0  | 1000.0  |
|  InvertedDoublePendulum  |   |  8370.0  | 8364.3  |   | 9337.5  | 9349.2  |

## Citing XuanCE ##
If you use XuanCE in your work, please cite our github repository:




<!-- More algorithms and documentations in detail are on the way.
More performance evaluations are on the way.
Some experiment results are shown below:
<p align="center">
<img src="./figures/cartpole.png"  width="400" height="300">
<img src="./figures/halfcheetah.png"  width="400" height="300">
</p>
![image](./figures/cartpole.png) -->


