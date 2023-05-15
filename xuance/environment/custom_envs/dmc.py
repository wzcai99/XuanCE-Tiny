from xuance.environment import *
from dm_control import suite
import gym.spaces
import cv2
class DMControl(gym.Env):
    def __init__(self,domain_name="humanoid",task_name="stand",timelimit=100,render_mode='rgb_array'):
        self.domain_name = domain_name
        self.task_name = task_name
        self.env = suite.load(domain_name=domain_name,task_name=task_name)
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        
        self.timelimit = timelimit
        self.render_mode = render_mode
        self.action_space = self.make_action_space(self.action_spec)
        self.observation_space = self.make_observation_space(self.observation_spec)
        self._metadata = {}
        self._reward_range = (-float("inf"), float("inf"))
        
    def make_observation_space(self,obs_spec):
        obs_shape_dim = 0
        for key,value in zip(obs_spec.keys(),obs_spec.values()):
            shape = value.shape
            if len(shape) == 0:
                obs_shape_dim += 1
            else:
                obs_shape_dim += shape[0]
        return gym.spaces.Box(-np.inf,-np.inf,(obs_shape_dim,))
            
    def make_action_space(self,act_spec):
        return gym.spaces.Box(act_spec.minimum,act_spec.maximum,act_spec.shape)
    
    def render(self):
        camera_frame0 = self.env.physics.render(camera_id=0, height=240, width=320)
        camera_frame1 = self.env.physics.render(camera_id=1, height=240, width=320)
        if self.render_mode == 'rgb_array':
            return np.concatenate((camera_frame0,camera_frame1),axis=0)
        elif self.render_mode == 'human':
            cv2.imshow("render_dmc",np.concatenate((camera_frame0,camera_frame1),axis=0))
            cv2.waitKey(5)
    
    def make_observation(self,timestep_data):
        return_observation = np.empty((0,),dtype=np.float32)
        for key,value in zip(timestep_data.observation.keys(),timestep_data.observation.values()):
            value = np.array([value],np.float32)
            if len(value.shape) == 1:
                return_observation=np.concatenate((return_observation,value),axis=-1)
            elif len(value.shape) == 2:
                return_observation=np.concatenate((return_observation,value[0]),axis=-1)
            else:
                raise NotImplementedError
        return return_observation
    
    def reset(self):
        self.episode_time = 0
        timestep_data = self.env.reset()
        info = {}
        return self.make_observation(timestep_data),info

    def step(self,action):
        timestep_data = self.env.step(action)
        next_obs = self.make_observation(timestep_data)
        reward = timestep_data.reward
        done = (self.episode_time >= self.timelimit)
        self.episode_time += 1
        trunction = (self.episode_time >= self.timelimit)
        info = {}
        return next_obs,reward,done,trunction,info
        
        
        
    
            
        
        
        
        