from xuance.learner import *
class DQN_Learner:
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
        self.logdir = config.logdir
        self.modeldir = config.modeldir
        self.update_frequency = config.update_frequency
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
        
        _,evalQ,_ = self.policy(input_batch)
        _,_,targetQ = self.policy(next_input_batch)
        
        evalQ = (evalQ * F.one_hot(tensor_action_batch.long(),evalQ.shape[-1])).sum(-1)
        targetQ = targetQ.max(dim=-1).values
        targetQ = tensor_reward_batch + self.gamma*(1-tensor_terminal_batch)*targetQ
        
        loss = F.mse_loss(evalQ,targetQ.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        if self.iterations % self.update_frequency == 0:
            self.policy.update_target()
        
        self.summary.add_scalar("q-loss",loss.item(),self.iterations)
        self.summary.add_scalar("learning-rate",self.optimizer.state_dict()['param_groups'][0]['lr'],self.iterations)
        self.summary.add_scalar("value_function",evalQ.mean().item(),self.iterations)
        
        if self.iterations % self.save_model_frequency == 0:
            time_string = time.asctime().replace(":", "_")#.replace(" ", "_")
            model_path = self.modeldir + "model-%s-%s.pth" % (time_string, str(self.iterations))
            torch.save(self.policy.state_dict(), model_path)
        
        