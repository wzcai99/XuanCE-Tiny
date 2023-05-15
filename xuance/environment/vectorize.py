from xuance.environment import *
# referenced from openai/baselines
class AlreadySteppingError(Exception):
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)
class NotSteppingError(Exception):
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

def tile_images(images):
    image_nums = len(images)
    image_shape = images[0].shape
    image_height = image_shape[0]
    image_width = image_shape[1]
    rows = (image_nums - 1) // 4 + 1
    if image_nums >= 4:
        cols = 4
    else:
        cols = image_nums
    try:
        big_img = np.zeros(
            (rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1), image_shape[2]), np.uint8)
    except IndexError:
        big_img = np.zeros((rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1)), np.uint8)
    for i in range(image_nums):
        c = i % 4
        r = i // 4
        big_img[10 * r + image_height * r:10 * r + image_height * r + image_height,
        10 * c + image_width * c:10 * c + image_width * c + image_width] = images[i]
    return big_img

class VecEnv(ABC):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False
    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass
    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass
    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass
    @abstractmethod
    def get_images(self):
        """
        Return RGB images from each environment
        """
        pass
    @abstractmethod
    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    def render(self, mode):
        imgs = self.get_images()
        big_img = tile_images(imgs)
        if mode == "human":
            cv2.imshow("render", cv2.cvtColor(big_img,cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return imgs
        else:
            raise NotImplementedError
    def close(self):
        if self.closed == True:
            return
        self.close_extras()
        self.closed = True

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, envs):
        self.waiting = False
        self.closed = False
        self.envs = envs
        env = self.envs[0]
        VecEnv.__init__(self, len(envs), env.observation_space, env.action_space)
        self.obs_shape = space2shape(self.observation_space)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                            zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        
    def reset(self):
        for e in range(self.num_envs):
            obs,info = self.envs[e].reset()
            self._save_obs(e, obs)
            self.buf_infos[e] = info
        return copy.deepcopy(self.buf_obs),self.buf_infos.copy()
    
    def step_async(self, actions):
        if self.waiting == True:
            raise AlreadySteppingError
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if listify == False:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]
        self.waiting = True
        
    def step_wait(self):
        if self.waiting == False:
            raise NotSteppingError
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_trunctions[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e] or self.buf_trunctions[e]:
                obs,_ = self.envs[e].reset()
            self._save_obs(e, obs)
        self.waiting = False
        return copy.deepcopy(self.buf_obs), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos.copy()
    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()
    def get_images(self):
        return [env.render() for env in self.envs]
    def render(self,mode):
        return super().render(mode)
    # save observation of indexes of e environment
    def _save_obs(self, e, obs):
        if isinstance(self.observation_space,gym.spaces.Dict):
            for k in self.obs_shape.keys():
                self.buf_obs[k][e] = obs[k]
        else:
            self.buf_obs[e] = obs

    