from environment import *
class BasicWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._reward_range = env._reward_range
        self._metadata = env._metadata
        if not isinstance(self.observation_space,gym.spaces.Dict):
            self._observation_space = gym.spaces.Dict({'observation':self._observation_space})
            
    def reset(self):
        self.episode_length = 0
        self.episode_score = 0 
        obs,info = super().reset()
        info['episode_length'] = self.episode_length
        info['episode_score'] = self.episode_score
        if isinstance(obs,dict):
            info['next_observation'] = obs
            return obs,info
        info['next_observation'] = {'observation':obs}
        return {'observation':obs},info
    
    def step(self,action):
        next_obs,reward,terminal,trunction,info = super().step(action)
        self.episode_length += 1
        self.episode_score += reward
        info['episode_length'] = self.episode_length
        info['episode_score'] = self.episode_score
        if isinstance(next_obs,dict):
            info['next_observation'] = next_obs
            return next_obs,reward,terminal,trunction,info
        info['next_observation'] = {'observation':next_obs}
        return {'observation':next_obs},reward,terminal,trunction,info

class NormActionWrapper(gym.ActionWrapper):
    def __init__(self,env,input_action_range=(-1,1)):
        super().__init__(env)
        self.input_action_range = input_action_range
        assert isinstance(self.action_space,gym.spaces.Box), "Only use the NormActionWrapper for Continuous Action."
    def action(self,act):
        act = np.clip(act,self.input_action_range[0],self.input_action_range[1])
        assert np.min(act) >= self.input_action_range[0] and np.max(act) <= self.input_action_range[1], "input action is out of the defined action range."
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high
        self.input_action_low = self.input_action_range[0]
        self.input_action_high = self.input_action_range[1]
        input_prop = (act - self.input_action_low) / (self.input_action_high - self.input_action_low)
        output_action = input_prop * (self.action_space_high - self.action_space_low) + self.action_space_low
        return output_action
    
class NormRewardWrapper(gym.RewardWrapper):
    def __init__(self,env,scale_reward_range=(0.01,100),clip_reward_range=(-5,5),gamma=0.99):
        super().__init__(env)
        assert scale_reward_range[0] < scale_reward_range[1], "invalid scale_reward_range."
        assert clip_reward_range[0] < clip_reward_range[1], "Invalid clip_reward_range."
        assert gamma < 1, "Gamma should be a float value smaller than 1."
        self.gamma = gamma
        self.scale_reward_range = scale_reward_range
        self.clip_reward_range = clip_reward_range
        self.return_rms = Running_MeanStd({'return':(1,)})
        self.episode_rewards = []
    def reward(self,reward):
        self.episode_rewards.append(reward)
        scale = np.clip(self.return_rms.std['return'][0],self.scale_reward_range[0],self.scale_reward_range[1])
        clip_reward = np.clip(reward/scale,self.clip_reward_range[0],self.clip_reward_range[1])
        return clip_reward
    def reset(self,**kwargs):
        if len(self.episode_rewards) > 0:
            self.return_rms.update({'return':discount_cumsum(self.episode_rewards,self.gamma)[0:1][np.newaxis,:]})
            self.episode_rewards.clear()
        return super().reset()
    
# ToDo: A more stable version for Norm observation      
class NormObservationWrapper(gym.ObservationWrapper):
    def __init__(self,env,scale_observation_range=(0.01,100),clip_observation_range=(-5,5)):
        super().__init__(env)
        self.scale_observation_range = scale_observation_range
        self.clip_observation_range = clip_observation_range
        self.observation_rms = Running_MeanStd(space2shape(env.observation_space))
        self._observation_rms = Running_MeanStd(space2shape(env.observation_space))
    def observation(self, observation):
        self.observation_rms.update(observation)
        norm_observation = {}
        for key,value in zip(observation.keys(),observation.values()):
            scale_factor = np.clip(1/(self.observation_rms.std[key] + 1e-7),self.scale_observation_range[0],self.scale_observation_range[1])
            norm_observation[key] = np.clip((value - self.observation_rms.mean[key]) * scale_factor,self.clip_observation_range[0],self.clip_observation_range[1])
        return norm_observation
  
