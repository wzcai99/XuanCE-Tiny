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
        
        self.train_episodes = np.zeros((self.nenvs,),np.int32)
        self.train_steps = 0
    
    def interact(self,inputs):
        outputs,dist,v = self.policy(inputs)
        action = dist.sample().detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        for key,value in zip(outputs.keys(),outputs.values()):
            outputs[key] = value.detach().cpu().numpy()        
        return outputs,action,v
    
    def train(self,train_steps:int=10000):
        obs,infos = self.environment.reset()
        for _ in tqdm(range(train_steps)):
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
                    approx_kl = self.learner.update(input_batch,action_batch,output_batch,return_batch,advantage_batch)
                    if approx_kl > self.config.target_kl:
                        break
                self.memory.clear()
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    self.train_episodes[i] += 1
                    self.summary.add_scalars("rewards-episode",{"env-%d"%i:infos[i]['episode_score']},self.train_episodes[i])
                    self.summary.add_scalars("rewards-steps",{"env-%d"%i:infos[i]['episode_score']},self.train_steps)
            obs = next_obs
            self.train_steps += 1

    def test(self,test_episode=10,render=False):
        import copy
        test_environment = copy.deepcopy(self.environment)
        obs,infos = test_environment.reset()
        current_episode = 0
        scores = []
        while current_episode < test_episode:
            if render:
                test_environment.render("human")
            outputs,actions,pred_values = self.interact(obs)
            next_obs,rewards,terminals,trunctions,infos = test_environment.step(actions)
            for i in range(self.nenvs):
                if terminals[i] == True or trunctions[i] == True:
                    scores.append(infos[i]['episode_score'])
                    current_episode += 1
            obs = next_obs
        return scores
    
    def benchmark(self,train_steps:int=10000,evaluate_steps:int=10000,test_episode=10,render=False):
        import time
        epoch = int(train_steps / evaluate_steps)
        benchmark_scores = []
        benchmark_scores.append({'steps':self.train_steps,'scores':self.test(test_episode,render)})
        for i in range(epoch):
            if i == epoch - 1:
                train_step = train_step - ((i+1)*evaluate_steps)
            else:
                train_step = evaluate_steps
            self.train(train_step)
            benchmark_scores.append({'steps':self.train_steps,'scores':self.test(test_episode,render)})
        time_string = time.asctime().replace(":", "_")#.replace(" ", "_")
        np.save(self.config.logdir+"benchmark_%s.npy"%time_string, benchmark_scores)

    

