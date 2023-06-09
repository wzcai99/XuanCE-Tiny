from xuance.agent import *
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
        
        self.logger = self.config.logger
        self.summary = self.learner.summary
        self.train_episodes = np.zeros((self.nenvs,),np.int32)
        self.train_steps = 0
        
        if self.logger=="wandb":
            wandb.define_metric("steps")
            wandb.define_metric("train-rewards/*",step_metric="steps")
            wandb.define_metric("evaluate-rewards/*",step_metric="steps")
    
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
        obs,infos = self.environment.reset()
        for _ in tqdm(range(train_steps)):
            outputs,actions = self.interact(obs,self.egreedy)
            next_obs,rewards,terminals,trunctions,infos = self.environment.step(actions)
            store_next_obs = next_obs.copy()
            
            for i in range(self.nenvs):
                if trunctions[i]:
                    for key in infos[i].keys():
                        if key in store_next_obs.keys():
                            store_next_obs[key][i] = infos[i][key]
            self.memory.store(obs,actions,outputs,rewards,terminals,store_next_obs)
            
            if self.memory.size >= self.start_training_size and self.train_steps % self.training_frequency == 0:
                input_batch,action_batch,output_batch,reward_batch,terminal_batch,next_input_batch = self.memory.sample()
                self.learner.update(input_batch,action_batch,reward_batch,terminal_batch,next_input_batch)
                        
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    self.train_episodes[i] += 1
                    if self.logger == "tensorboard":
                        self.summary.add_scalars("train-rewards-steps",{"env-%d"%i:infos[i]['episode_score']},self.train_steps*self.nenvs)
                    else:
                         wandb.log({f"train-rewards/{i}":infos[i]['episode_score'],'steps':self.train_steps*self.nenvs})

            
            obs = next_obs
            self.train_steps += 1
            self.egreedy = self.egreedy - (self.start_egreedy-self.end_egreedy)/self.config.train_steps
    
    def test(self,test_environment,test_episode=10,render=False):
        obs,infos = test_environment.reset()
        current_episode = 0
        scores = []
        images = [[] for i in range(test_environment.num_envs)]
        best_score = -np.inf
       
        while current_episode < test_episode:
            if render:
                test_environment.render("human")
            else:
                render_images = test_environment.render('rgb_array')
                for index,img in enumerate(render_images):
                    images[index].append(img)
            outputs,actions = self.interact(obs,0)
            next_obs,rewards,terminals,trunctions,infos = test_environment.step(actions)
            for i in range(test_environment.num_envs):
                if terminals[i] == True or trunctions[i] == True:
                    if self.logger == 'tensorboard':
                        self.summary.add_scalars("evaluate-score",{"episode-%d"%current_episode:infos[i]['episode_score']},self.train_steps*self.nenvs)
                    else:
                        wandb.log({f"evaluate-rewards/{current_episode}":infos[i]['episode_score'],'steps':self.train_steps*self.nenvs})
                    
                    if infos[i]['episode_score'] > best_score:
                        episode_images = images[i].copy()
                        best_score = infos[i]['episode_score']
                        
                    scores.append(infos[i]['episode_score'])
                    images[i] = []
                    current_episode += 1
            obs = next_obs
            
        if self.logger == "wandb":
            wandb.log({"video":wandb.Video(np.array(episode_images[0],dtype=np.uint8).transpose(0,3,1,2),fps=50,format='gif'),
                    'train-step':self.train_steps*self.nenvs})
        print("[%s] Training Steps:%.2f K, Evaluate Episodes:%d, Score Average:%f, Std:%f"%(get_time_hm(), self.train_steps*self.nenvs/1000,
                                                                                       test_episode,np.mean(scores),np.std(scores)))
        return scores,episode_images
    
    def benchmark(self,test_environment,train_steps:int=10000,evaluate_steps:int=10000,test_episode=10,render=False,save_best_model=True):
        import time
        epoch = int(train_steps / evaluate_steps) + 1

        evaluate_scores,evaluate_video = self.test(test_environment,test_episode,render)
        benchmark_scores = []
        benchmark_scores.append({'steps':self.train_steps,'scores':evaluate_scores})
       
        
        best_average_score = np.mean(benchmark_scores[-1]['scores'])
        best_std_score = np.std(benchmark_scores[-1]['scores'])
        best_video = evaluate_video
        
        for i in range(epoch):
            if i == epoch - 1:
                train_step = train_steps - (i*evaluate_steps)
            else:
                train_step = evaluate_steps
            self.train(train_step)
           
            evaluate_scores,evaluate_video = self.test(test_environment,test_episode,render)
            benchmark_scores.append({'steps':self.train_steps,'scores':evaluate_scores})

            
            if np.mean(benchmark_scores[-1]['scores']) > best_average_score:
                best_average_score = np.mean(benchmark_scores[-1]['scores'])
                best_std_score = np.std(benchmark_scores[-1]['scores'])
                best_video = evaluate_video
                if save_best_model == True:
                    model_path = self.learner.modeldir + "/best_model.pth"
                    torch.save(self.policy.state_dict(), model_path)
        
        if self.logger == "tensorboard":
            self.summary.add_video("video",torch.as_tensor(np.array(best_video,dtype=np.uint8).transpose(0,3,1,2),dtype=torch.uint8).unsqueeze(0),fps=50,global_step=self.nenvs*self.train_steps)
        else:
            wandb.log({"video":wandb.Video(np.array(best_video,dtype=np.uint8).transpose(0,3,1,2),fps=50,format='gif')},commit=False)
            
        np.save(self.learner.logdir+"/benchmark_%s.npy"%get_time_full(), benchmark_scores)
        print("Best Model score = %f, std = %f"%(best_average_score,best_std_score))
             