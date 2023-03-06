from responses import target
from learner import *
class DDPG_Learner:
    def __init__(self,
                 config,
                 policy,
                 optimizer,
                 scheduler,
                 device):
        self.policy = policy
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.actor_scheduler = scheduler[0]
        self.critic_scheduler = scheduler[1]
        self.device = device
        
        self.tau = config.tau
        self.gamma = config.gamma
        
        self.logdir = config.logdir
        self.modeldir = config.modeldir
        self.save_model_frequency = config.save_model_frequency
        self.summary = SummaryWriter(self.logdir)
        self.iterations = 0
        create_directory(self.logdir)
        create_directory(self.modeldir)
        
    def update(self,input_batch,action_batch,reward_batch,terminal_batch,next_input_batch):
        self.iterations += 1
        tensor_action_batch = torch.as_tensor(action_batch,device=self.device)
        tensor_reward_batch = torch.as_tensor(reward_batch,device=self.device)
        tensor_terminal_batch = torch.as_tensor(terminal_batch,device=self.device)
        
        #update Q network
        with torch.no_grad():
            targetQ = self.policy.Qtarget(next_input_batch)
            targetQ = tensor_reward_batch + self.gamma * (1-tensor_terminal_batch) * targetQ
            
        currentQ = self.policy.Qaction(input_batch,tensor_action_batch)
        Q_loss = F.mse_loss(currentQ,targetQ)
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        # update A network
        _,_,evalQ = self.policy(input_batch)
        A_loss = -evalQ.mean()
        self.actor_optimizer.zero_grad()
        A_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.policy.soft_update(self.tau)
        
        self.summary.add_scalar("Q-loss",Q_loss.item(),self.iterations)
        self.summary.add_scalar("A-loss",A_loss.item(),self.iterations)
        self.summary.add_scalar("actor-learning-rate",self.actor_optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
        self.summary.add_scalar("critic-learning-rate",self.critic_optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
        self.summary.add_scalar("value_function",evalQ.mean().item(),self.iterations)
        
        if self.iterations % self.save_model_frequency == 0:
            time_string = time.asctime()
            time_string = time_string.replace(" ", "")
            model_path = self.modeldir + "model-%s-%s.pth" % (time.asctime(), str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)
        