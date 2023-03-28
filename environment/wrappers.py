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
    
  
