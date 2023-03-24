from policy import *
class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 initialize,
                 device):
        super(BasicQhead,self).__init__()
        self.model = nn.Sequential(*mlp_block(state_dim,action_dim,None,initialize,device)[0])
    def forward(self,x:torch.Tensor):
        return self.model(x)

class DuelQhead(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 initialize,
                 device):
        super(DuelQhead,self).__init__()
        self.a_model = nn.Sequential(*mlp_block(state_dim,action_dim,None,initialize,device)[0])
        self.v_model = nn.Sequential(*mlp_block(state_dim,1,None,initialize,device)[0])
    def forward(self,x:torch.Tensor):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - a.mean(dim=-1).unsqueeze(dim=-1))
        return q
        
class DQN_Policy(nn.Module):
    def __init__(self,
                 action_space,
                 representation,
                 initialize,
                 device):
        super(DQN_Policy,self).__init__()
        assert isinstance(action_space,gym.spaces.Discrete), "DQN is not supported for non-discrete action space"
        self.action_dim = action_space.n
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        self.output_shape['evalQ'] = (self.action_dim,)
        self.output_shape['targetQ'] = (self.action_dim,)
        self.eval_representation = representation
        self.evalQ = BasicQhead(representation.output_shape['state'][0],self.action_dim,initialize,device)
        self.target_representation = copy.deepcopy(self.eval_representation)
        self.targetQ = copy.deepcopy(self.evalQ)
    def forward(self,observation:dict):
        eval_outputs = self.eval_representation(observation)
        target_outputs = self.target_representation(observation)
        evalQ = self.evalQ(eval_outputs['state'])
        targetQ = self.targetQ(target_outputs['state']).detach()
        eval_outputs['evalQ'] = evalQ
        eval_outputs['targetQ'] = targetQ
        return eval_outputs,evalQ,targetQ
    def update_target(self):
        for ep,tp in zip(self.eval_representation.parameters(),self.target_representation.parameters()):
            tp.data.copy_(ep.data)
        for ep,tp in zip(self.evalQ.parameters(),self.targetQ.parameters()):
            tp.data.copy_(ep.data)
    
class DuelDQN_Policy(DQN_Policy):
    def __init__(self,
                 action_space,
                 representation,
                 initialize,
                 device):
        super(DuelDQN_Policy,self).__init__(action_space,representation,initialize,device)
        assert isinstance(action_space,gym.spaces.Discrete), "DQN is not supported for non-discrete action space"
        self.evalQ = DuelQhead(representation.output_shape['state'][0],self.action_dim,initialize,device)
        self.targetQ = copy.deepcopy(self.evalQ)
        
  


    