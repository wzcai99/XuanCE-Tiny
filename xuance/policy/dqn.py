from xuance.policy import *
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
    
class C51Qhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(C51Qhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        self.model = nn.Sequential(*mlp_block(state_dim,action_dim*atom_num,None,initialize,device)[0])
    def forward(self, x: torch.Tensor):
        dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
        dist_probs = F.softmax(dist_logits, dim=-1)
        return dist_probs

class QRDQNhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QRDQNhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        self.model = nn.Sequential(*mlp_block(state_dim,action_dim*atom_num,None,initialize,device)[0])
    def forward(self, x: torch.Tensor):
        quantiles = self.model(x).view(-1, self.action_dim, self.atom_num)
        return quantiles
        
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
        assert isinstance(action_space,gym.spaces.Discrete), "Dueling-DQN is not supported for non-discrete action space"
        self.evalQ = DuelQhead(representation.output_shape['state'][0],self.action_dim,initialize,device)
        self.targetQ = copy.deepcopy(self.evalQ)
        
class C51_Policy(DQN_Policy):
    def __init__(self,
                 action_space,
                 representation,
                 value_range,
                 atom_num,
                 initialize,
                 device):
        super(C51_Policy,self).__init__(action_space,representation,initialize,device)
        assert isinstance(action_space,gym.spaces.Discrete), "C51 is not supported for non-discrete action space"
        self.evalQ = C51Qhead(representation.output_shape['state'][0],action_space.n,atom_num,initialize,device)
        self.targetQ = copy.deepcopy(self.evalQ)
        self.value_range = value_range
        self.atom_num = atom_num
        self.supports = torch.nn.Parameter(torch.linspace(self.value_range[0], self.value_range[1], self.atom_num), requires_grad=False).to(device)
        self.deltaz = (value_range[1] - value_range[0]) / (atom_num - 1)
    
class QRDQN_Policy(DQN_Policy):
    def __init__(self,
                 action_space,
                 representation,
                 atom_num,
                 initialize,
                 device):
        super(QRDQN_Policy,self).__init__(action_space,representation,initialize,device)
        assert isinstance(action_space,gym.spaces.Discrete), "QRDQN is not supported for non-discrete action space"
        self.evalQ = QRDQNhead(representation.output_shape['state'][0],action_space.n,atom_num,initialize,device)
        self.targetQ = copy.deepcopy(self.evalQ)
        self.atom_num = atom_num
        
        
  


    