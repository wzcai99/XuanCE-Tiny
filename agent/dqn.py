from agent import *
class DQN_Agent:
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
        self.minibatch = config.minibatch
        self.gamma = config.gamma
        self.input_shape = self.policy.input_shape
        self.action_shape = self.environment.action_space.shape
        self.output_shape = self.policy.output_shape
        self.start_egreedy = config.start_egreedy 
        self.end_egreedy = config.end_egreedy
        self.egreedy = self.start_egreedy

        self.start_training_size = config.start_training_size
        self.training_frequency = config.training_frequency
        self.memory = DummyOffPolicyBuffer(self.input_shape,
                                           self.action_shape,
                                           self.output_shape,
                                           self.nenvs,
                                           self.nsize,
                                           self.minibatch)
        self.summary = SummaryWriter(self.config.logdir)
    
    def interact(self,inputs,egreedy):
        outputs,evalQ,_ = self.policy(inputs)
        argmax_action = evalQ.argmax(dim=-1)
        random_action = np.random.choice(self.environment.action_space.n,self.nenvs)
        if np.random.rand() < egreedy:
            action = random_action
        else:
            action = argmax_action.detach().cpu().numpy()
        for key,value in zip(outputs.keys(),outputs.values()):
            outputs[key] = value.detach().cpu().numpy()        
        return outputs,action
    
    def train(self,train_steps:int=10000):
        episodes = np.zeros((self.nenvs,),np.int32)
        obs,infos = self.environment.reset()
        for step in tqdm(range(train_steps)):
            outputs,actions = self.interact(obs,self.egreedy)
            next_obs,rewards,terminals,trunctions,infos = self.environment.step(actions)
            store_next_obs = next_obs.copy()
            
            for i in range(self.nenvs):
                if trunctions[i]:
                    for key in infos[i].keys():
                        if key in store_next_obs.keys():
                            store_next_obs[key][i] = infos[i][key]
            self.memory.store(obs,actions,outputs,rewards,terminals,store_next_obs)
            
            if self.memory.size >= self.start_training_size and step % self.training_frequency == 0:
                input_batch,action_batch,output_batch,reward_batch,terminal_batch,next_input_batch = self.memory.sample()
                self.learner.update(input_batch,action_batch,reward_batch,terminal_batch,next_input_batch)
                        
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    episodes[i] += 1
                    self.summary.add_scalars("rewards-episode",{"env-%d"%i:infos[i]['episode_score']},episodes[i])
                    self.summary.add_scalars("rewards-steps",{"env-%d"%i:infos[i]['episode_score']},step)
            
            obs = next_obs
            self.egreedy = self.egreedy - (self.start_egreedy-self.end_egreedy)/train_steps
            
    
    def test(self,model_path,test_steps = 10000,egreedy = 0.05):
        self.policy.load_state_dict(torch.load(self.config.modeldir + model_path))
        obs,infos = self.environment.reset()
        for step in tqdm(range(test_steps)):
            outputs,actions = self.interact(obs,egreedy)
            next_obs,rewards,terminals,trunctions,infos = self.environment.step(actions)
            obs = next_obs
            
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    print(infos[i]['episode_score'])
            
        
             