import gym
import numpy as np
from xuance.environment import *

class EnvPool_Wrapper:
    def __init__(self,vecenv):
        self.vecenv = vecenv
        self.num_envs = vecenv.config['num_envs']
        if isinstance(vecenv.observation_space,gym.spaces.Dict):
            self.observation_space = vecenv.observation_space
        else:
            self.observation_space = gym.spaces.Dict({'observation':vecenv.observation_space})
        self.action_space = vecenv.action_space
    
    def reset(self):
        obs,_ = self.vecenv.reset()
        self.episode_lengths = np.zeros((self.num_envs,),np.int32)
        self.episode_scores = np.zeros((self.num_envs,),np.float32)
        self.last_episode_lengths = self.episode_lengths.copy()
        self.last_episode_scores = self.episode_scores.copy()
        infos = []
        for i in range(self.num_envs):
            if isinstance(obs,dict):
                current_dict = {}
                for key,value in zip(obs.keys(),obs.values()):
                    current_dict[key] = value[i]
                infos.append({'episode_length':self.episode_lengths[i],
                              'episode_score':self.episode_scores[i],
                              'next_observation':current_dict})
            else:
                infos.append({'episode_length':self.episode_lengths[i],
                              'episode_score':self.episode_scores[i],
                              'next_observation':{'observation':obs[i]}})
        if isinstance(obs,dict):
            return obs,infos
        else:
            return {'observation':obs},infos
    
    def step(self,actions):
        self.last_episode_lengths = self.episode_lengths.copy()
        self.last_episode_scores = self.episode_scores.copy()
        obs,rewards,terminals,trunctions,_ = self.vecenv.step(actions)
        self.episode_lengths += 1
        self.episode_scores += rewards
        infos = []
        for i in range(self.num_envs):
            if terminals[i] or trunctions[i]:
                self.episode_scores[i] = 0
                self.episode_lengths[i] = 0
            if isinstance(obs,dict):
                current_dict = {}
                for key,value in zip(obs.keys(),obs.values()):
                    current_dict[key] = value[i]
                infos.append({'episode_length':self.last_episode_lengths[i],
                              'episode_score':self.last_episode_scores[i],
                              'next_observation':current_dict})
            else:
                infos.append({'episode_length':self.last_episode_lengths[i],
                              'episode_score':self.last_episode_scores[i],
                              'next_observation':{'observation':obs[i]}})
        if isinstance(obs,dict):
            return obs,rewards,terminals,trunctions,infos         
        else:
            return {'observation':obs},rewards,terminals,trunctions,infos
        
        
class EnvPool_Normalizer:
    def __init__(self,vecenv:EnvPool_Wrapper):
        self.vecenv = vecenv
        self.num_envs = vecenv.num_envs
        self.observation_space = self.vecenv.observation_space
        self.action_space = self.vecenv.action_space
    def step(self,actions):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class EnvPool_ObservationNorm(EnvPool_Normalizer):
    def __init__(self,config,vecenv,scale_range=(0.1,10),obs_range=(-5,5),forbidden_keys=[],train=True):
        super(EnvPool_ObservationNorm,self).__init__(vecenv)
        assert scale_range[0] < scale_range[1], "invalid scale_range."
        assert obs_range[0] < obs_range[1], "Invalid reward_range."
        self.config = config
        self.scale_range = scale_range
        self.obs_range = obs_range
        self.forbidden_keys = forbidden_keys
        self.obs_rms = Running_MeanStd(space2shape(self.observation_space))
        self.train = train
        self.save_dir = os.path.join(self.config.modeldir,self.config.env_name,self.config.algo_name+"-%d"%self.config.seed)
        if self.train == False:
            self.load_rms()
            
    def load_rms(self):
        npy_path = os.path.join(self.save_dir,"observation_stat.npy")
        if not os.path.exists(npy_path):
            return
        rms_data = np.load(npy_path,allow_pickle=True).item()
        self.obs_rms.count = rms_data['count']
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
    
    def reset(self):
        self.train_steps = 0
        return self.vecenv.reset()
    
    def step(self,actions):
        self.train_steps += 1
        if self.config.train_steps == self.train_steps or self.train_steps % self.config.save_model_frequency == 0:
            np.save(os.path.join(self.save_dir,"observation_stat.npy"),{'count':self.obs_rms.count,'mean':self.obs_rms.mean,'var':self.obs_rms.var})
        obs,rews,terminals,trunctions,infos = self.vecenv.step(actions)
        if self.train:
            self.obs_rms.update(obs)
        norm_observation = {}
        for key,value in zip(obs.keys(),obs.values()):
            if key in self.forbidden_keys:
                continue
            scale_factor = np.clip(1/(self.obs_rms.std[key] + 1e-7),self.scale_range[0],self.scale_range[1])
            norm_observation[key] = np.clip((value - self.obs_rms.mean[key]) * scale_factor,self.obs_range[0],self.obs_range[1])
        return norm_observation,rews,terminals,trunctions,infos
            
class EnvPool_RewardNorm(EnvPool_Normalizer):
    def __init__(self,config,vecenv,scale_range=(0.1,10),reward_range=(-5,5),gamma=0.99,train=True):
        super(EnvPool_RewardNorm,self).__init__(vecenv)
        assert scale_range[0] < scale_range[1], "invalid scale_range."
        assert reward_range[0] < reward_range[1], "Invalid reward_range."
        assert gamma < 1, "Gamma should be a float value smaller than 1."
        self.config = config
        self.gamma = gamma
        self.scale_range = scale_range
        self.reward_range = reward_range
        self.return_rms = Running_MeanStd({'return':(1,)})
        self.episode_rewards = [[] for i in range(self.num_envs)]
        self.train = train
        self.save_dir = os.path.join(self.config.modeldir,self.config.env_name,self.config.algo_name+"-%d"%self.config.seed)
        if train == False:
            self.load_rms()
    def load_rms(self):
        npy_path = os.path.join(self.save_dir,"reward_stat.npy")
        if not os.path.exists(npy_path):
            return
        rms_data = np.load(npy_path,allow_pickle=True).item()
        self.return_rms.count = rms_data['count']
        self.return_rms.mean = rms_data['mean']
        self.return_rms.var = rms_data['var']
    
    def reset(self):
        self.train_steps = 0
        return self.vecenv.reset()
    def step(self,act):
        self.train_steps += 1
        if self.config.train_steps == self.train_steps or self.train_steps % self.config.save_model_frequency == 0:
            np.save(os.path.join(self.save_dir,"reward_stat.npy"),{'count':self.return_rms.count,'mean':self.return_rms.mean,'var':self.return_rms.var})
        obs,rews,terminals,trunctions,infos = self.vecenv.step(act)
        for i in range(len(rews)):
            if terminals[i] != True and trunctions[i] != True:
                self.episode_rewards[i].append(rews[i])
            else:
                if self.train:
                    self.return_rms.update({'return':discount_cumsum(self.episode_rewards[i],self.gamma)[0:1][np.newaxis,:]})
                self.episode_rewards[i].clear()
            scale = np.clip(self.return_rms.std['return'][0],self.scale_range[0],self.scale_range[1])
            rews[i] =  np.clip(rews[i]/scale,self.reward_range[0],self.reward_range[1])
        return obs,rews,terminals,trunctions,infos

class EnvPool_ActionNorm(EnvPool_Normalizer):
    def __init__(self,vecenv,input_action_range=(-1,1)):
        super(EnvPool_ActionNorm,self).__init__(vecenv)
        self.input_action_range = input_action_range
        assert isinstance(self.action_space,gym.spaces.Box), "Only use the NormActionWrapper for Continuous Action."
    def reset(self):
        return self.vecenv.reset()
    def step(self,actions):
        act = np.clip(actions,self.input_action_range[0],self.input_action_range[1])
        assert np.min(act) >= self.input_action_range[0] and np.max(act) <= self.input_action_range[1], "input action is out of the defined action range."
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high
        self.input_action_low = self.input_action_range[0]
        self.input_action_high = self.input_action_range[1]
        input_prop = (act - self.input_action_low) / (self.input_action_high - self.input_action_low)
        output_action = input_prop * (self.action_space_high - self.action_space_low) + self.action_space_low
        return self.vecenv.step(output_action)
    
    
        