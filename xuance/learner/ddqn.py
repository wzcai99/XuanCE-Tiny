from xuance.learner import *
class DDQN_Learner:
    def __init__(self,
                 config,
                 policy,
                 optimizer,
                 scheduler,
                 device):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gamma = config.gamma
        self.update_frequency = config.update_frequency
        self.save_model_frequency = config.save_model_frequency
        self.iterations = 0
        
        self.logdir = os.path.join(config.logdir,config.env_name,config.algo_name+"-%d"%config.seed)
        self.modeldir = os.path.join(config.modeldir,config.env_name,config.algo_name+"-%d/"%config.seed)
        self.logger = config.logger
        create_directory(self.modeldir)
        create_directory(self.logdir)
        if self.logger == 'wandb':
            self.summary = wandb.init(project="XuanCE",
                                      group=config.env_name,
                                      name=config.algo_name,
                                      config=wandb.helper.parse_config(vars(config), exclude=('logger','logdir','modeldir')))
        elif self.logger == 'tensorboard':
            self.summary = SummaryWriter(self.logdir)
        else:
            raise NotImplementedError  
        
    def update(self,input_batch,action_batch,reward_batch,terminal_batch,next_input_batch):
        self.iterations += 1
        tensor_action_batch = torch.as_tensor(action_batch,device=self.device)
        tensor_reward_batch = torch.as_tensor(reward_batch,device=self.device)
        tensor_terminal_batch = torch.as_tensor(terminal_batch,device=self.device)
        
        _,evalQ,_ = self.policy(input_batch)
        _,_,targetQ = self.policy(next_input_batch)
        _,next_evalQ,_ = self.policy(next_input_batch)
        
        evalQ = (evalQ * F.one_hot(tensor_action_batch.long(),evalQ.shape[-1])).sum(-1)
        targetA = next_evalQ.argmax(dim=-1)
        targetQ = (targetQ * F.one_hot(targetA.long(),targetQ.shape[-1])).sum(-1)
        targetQ = tensor_reward_batch + self.gamma*(1-tensor_terminal_batch)*targetQ
        
        loss = F.mse_loss(evalQ,targetQ.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        if self.iterations % self.update_frequency == 0:
            self.policy.update_target()
        
        if self.logger == 'tensorboard':
            self.summary.add_scalar("Q-loss",loss.item(),self.iterations)
            self.summary.add_scalar("learning-rate",self.optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
            self.summary.add_scalar("value_function",evalQ.mean().item(),self.iterations)
        else:
            wandb.log({'Q-loss':loss.item(),
                       "learning-rate":self.optimizer.state_dict()['param_groups'][0]['lr'],
                       "value_function":evalQ.mean().item(),
                       "iterations":self.iterations})
        
        if self.iterations % self.save_model_frequency == 0:
            time_string = time.asctime()
            time_string = time_string.replace(" ", "")
            model_path = self.modeldir + "model-%s-%s.pth" % (time.asctime(), str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)
        
        