from agent import *
class PPO_Agent:
    def __init__(self,
                 config,
                 environment,
                 policy,
                 learner):
        self.config = config
        self.environment = environment
        self.policy = policy
        self.learner = learner
        self.nenvs = environment.num_envs
        self.nsize = config.nsize
        self.nminibatch = config.nminibatch
        self.nepoch = config.nepoch
        self.gamma = config.gamma
        self.tdlam = config.tdlam
        self.input_shape = self.policy.input_shape
        self.action_shape = self.environment.action_space.shape
        self.output_shape = self.policy.output_shape
        self.memory = DummyOnPolicyBuffer(self.input_shape,
                                          self.action_shape,
                                          self.output_shape,
                                          self.nenvs,
                                          self.nsize,
                                          self.nminibatch,
                                          self.gamma,
                                          self.tdlam)
        self.summary = SummaryWriter(self.config.logdir)
    
    def interact(self,inputs):
        outputs,dist,v = self.policy(inputs)
        action = dist.sample().detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        for key,value in zip(outputs.keys(),outputs.values()):
            outputs[key] = value.detach().cpu().numpy()        
        return outputs,action,v
    
    def train(self,train_steps:int=10000):
        episodes = np.zeros((self.nenvs,),np.int32)
        obs,infos = self.environment.reset()
        for step in tqdm(range(train_steps)):
            outputs,actions,pred_values = self.interact(obs)
            next_obs,rewards,terminals,trunctions,infos = self.environment.step(actions)
            self.memory.store(obs,actions,outputs,rewards,pred_values)
            for i in range(self.nenvs):
                if terminals[i] == True:
                    self.memory.finish_path(0,i)
                elif trunctions[i] == True:
                    real_next_observation = infos[i]['next_observation']
                    for key in real_next_observation.keys():
                        real_next_observation[key] = real_next_observation[key][np.newaxis,:]
                    _,_,truncate_value = self.interact(real_next_observation)
                    self.memory.finish_path(truncate_value[0],i)
                    
            if self.memory.full:
                _,_,next_pred_values = self.interact(next_obs)
                for i in range(self.nenvs):
                    self.memory.finish_path(next_pred_values[i]*(1-terminals[i]),i)
                for _ in range(self.nminibatch * self.nepoch):
                    input_batch,action_batch,output_batch,return_batch,advantage_batch = self.memory.sample()
                    self.learner.update(input_batch,action_batch,output_batch,return_batch,advantage_batch)
                self.memory.clear()
                    
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    episodes[i] += 1
                    self.summary.add_scalars("rewards-episode",{"env-%d"%i:infos[i]['episode_score']},episodes[i])
                    self.summary.add_scalars("rewards-steps",{"env-%d"%i:infos[i]['episode_score']},step)
            
            obs = next_obs

    
    # def test()
    
    # def evaluate()
        
    

