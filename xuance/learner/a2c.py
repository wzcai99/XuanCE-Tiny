from xuance.learner import *
class A2C_Learner:
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
        
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clipgrad_norm = config.clipgrad_norm
        self.save_model_frequency = config.save_model_frequency
        self.iterations = 0
        self.logdir = os.path.join(config.logdir,config.env_name,config.algo_name+"-%d/"%config.seed)
        self.modeldir = os.path.join(config.modeldir,config.env_name,config.algo_name+"-%d/"%config.seed)
        self.logger = config.logger
        create_directory(self.modeldir)
        create_directory(self.logdir)
        
        if self.logger == 'wandb':
            self.summary = wandb.init(project="XuanCE",
                                      group=config.env_name,
                                      name=config.algo_name,
                                      config=wandb.helper.parse_config(vars(config), exclude=('logger','logdir','modeldir')))
            wandb.define_metric("iterations")
            wandb.define_metric("train/*",step_metric="iterations")
        elif self.logger == 'tensorboard':
            self.summary = SummaryWriter(self.logdir)
        else:
            raise NotImplementedError    
    

    def update(self,input_batch,action_batch,return_batch,advantage_batch):
        self.iterations += 1
        tensor_action_batch = torch.as_tensor(action_batch,device=self.device)
        tensor_return_batch = torch.as_tensor(return_batch,device=self.device)
        tensor_advantage_batch = torch.as_tensor(advantage_batch,device=self.device)
        
        _,actor,critic = self.policy(input_batch)
        actor_loss = -(tensor_advantage_batch * actor.logprob(tensor_action_batch)).mean()
        critic_loss = F.mse_loss(critic,tensor_return_batch)
        entropy_loss = actor.entropy().mean()
        
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clipgrad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        if self.logger == 'tensorboard':
            self.summary.add_scalar("actor-loss",actor_loss.item(),self.iterations)
            self.summary.add_scalar("critic-loss",critic_loss.item(),self.iterations)
            self.summary.add_scalar("entropy-loss",entropy_loss.item(),self.iterations)
            self.summary.add_scalar("learning-rate",self.optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
            self.summary.add_scalar("value_function",critic.mean().item(),self.iterations)
        else:
            wandb.log({'train/actor-loss':actor_loss.item(),
                       "train/critic-loss":critic_loss.item(),
                       "train/entropy-loss":entropy_loss.item(),
                       "train/learning-rate":self.optimizer.state_dict()['param_groups'][0]['lr'],
                       "train/value_function":critic.mean().item(),
                       "iterations":self.iterations})
        
        if self.iterations % self.save_model_frequency == 0:
            model_path = self.modeldir + "model-%s-%s.pth" % (get_time_full(), str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)