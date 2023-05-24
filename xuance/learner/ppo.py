from xuance.learner import *
class PPO_Learner:
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
        self.clip_range = config.clip_range
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
        
    def update(self,input_batch,action_batch,output_batch,return_batch,advantage_batch):
        self.iterations += 1
        tensor_action_batch = torch.as_tensor(action_batch,device=self.device)
        tensor_return_batch = torch.as_tensor(return_batch,device=self.device)
        tensor_advantage_batch = torch.as_tensor(advantage_batch,device=self.device)
        
        # get current policy distribution
        _,actor,critic = self.policy(input_batch)
        current_logp = actor.logprob(tensor_action_batch)
        # get old policy distribution
        _,old_actor,_ = self.policy(input_batch)
        param_dict = {}
        for key in self.policy.actor.output_shape.keys():
            param_dict[key] = torch.as_tensor(output_batch[key],device=self.device)
        old_actor.set_param(**param_dict)
        old_logp = old_actor.logprob(tensor_action_batch).detach()
        ratio = (current_logp - old_logp).exp().float()
        approx_kl = actor.kl_divergence(old_actor).mean()
        
        surrogate1 = tensor_advantage_batch * ratio
        surrogate2 = ratio.clamp(1-self.clip_range,1+self.clip_range)*tensor_advantage_batch
        
        actor_loss = -torch.minimum(surrogate1,surrogate2).mean()
        critic_loss = F.mse_loss(critic,tensor_return_batch)
        entropy_loss = actor.entropy().mean()
        
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clipgrad_norm)
        self.optimizer.step()
        self.scheduler.step()
        if self.logger == 'tensorboard':
            self.summary.add_scalar("ratio",ratio.mean().item(),self.iterations)
            self.summary.add_scalar("actor-loss",actor_loss.item(),self.iterations)
            self.summary.add_scalar("critic-loss",critic_loss.item(),self.iterations)
            self.summary.add_scalar("entropy-loss",entropy_loss.item(),self.iterations)
            self.summary.add_scalar("kl-divergence",approx_kl.item(),self.iterations)
            self.summary.add_scalar("learning-rate",self.optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
            self.summary.add_scalar("value_function",critic.mean().item(),self.iterations)
        else:
            wandb.log({'train/ratio':ratio.mean().item(),
                       'train/actor-loss':actor_loss.item(),
                       "train/critic-loss":critic_loss.item(),
                       "train/entropy-loss":entropy_loss.item(),
                       "train/kl-divergence":approx_kl.item(),
                       "train/learning-rate":self.optimizer.state_dict()['param_groups'][0]['lr'],
                       "train/value_function":critic.mean().item(),
                       "iterations":self.iterations})
        if self.iterations % self.save_model_frequency == 0:
            time_string = time.asctime().replace(":", "_")#.replace(" ", "_")
            model_path = self.modeldir + "model-%s-%s.pth" % (time_string, str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)
        return approx_kl
        
        