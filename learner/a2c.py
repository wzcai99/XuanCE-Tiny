from learner import *

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
        self.logdir = config.logdir
        self.modeldir = config.modeldir
        self.save_model_frequency = config.save_model_frequency
        self.summary = SummaryWriter(self.logdir)
        self.iterations = 0
        create_directory(self.logdir)
        create_directory(self.modeldir)
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
        
        self.summary.add_scalar("actor-loss",actor_loss.item(),self.iterations)
        self.summary.add_scalar("critic-loss",critic_loss.item(),self.iterations)
        self.summary.add_scalar("entropy-loss",entropy_loss.item(),self.iterations)
        self.summary.add_scalar("learning-rate",self.optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
        self.summary.add_scalar("value_function",critic.mean().item(),self.iterations)
        
        if self.iterations % self.save_model_frequency == 0:
            time_string = time.asctime()
            time_string = time_string.replace(" ", "")
            model_path = self.modeldir + "model-%s-%s.pth" % (time.asctime(), str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)
        