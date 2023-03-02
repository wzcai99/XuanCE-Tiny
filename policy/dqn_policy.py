from policy import *
from utils.layer import mlp_block
class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 activation,
                 initialize,
                 device):
        super(BasicQhead,self).__init__()
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),action_dim,None,initialize,device)
        self.model = nn.Sequential(*block1,*block2)
    def forward(self,x:torch.Tensor):
        return self.model(x)

class DuelQhead(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 activation,
                 initialize,
                 device):
        super(DuelQhead,self).__init__()
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),action_dim,None,initialize,device)
        self.a_model = nn.Sequential(*block1,*block2)
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),1,None,initialize,device)
        self.v_model = nn.Sequential(*block1,*block2)
    def forward(self,x:torch.Tensor):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - a.mean(dim=-1).unsqueeze(dim=-1))
        return q
    
class C51Qhead(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 atom_num:int,
                 activation,
                 initialize,
                 device):
        super(C51Qhead,self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),action_dim*atom_num,None,initialize,device)
        self.model = nn.Sequential(*block1,*block2)
    def forward(self,x:torch.Tensor):
        dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
        dist_probs = F.softmax(dist_logits, dim=-1)
        return dist_probs
    
class QRDQNhead(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 quantile_num,
                 activation,
                 initialize,
                 device):
        super(QRDQNhead,self).__init__()
        self.action_dim = action_dim
        self.quantile_num = quantile_num
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),action_dim*quantile_num,None,initialize,device)
        self.model = nn.Sequential(*block1,*block2)    
    def forward(self,x:torch.Tensor):
        quantiles = self.model(x).view(-1,self.action_dim,self.quantile_num)
        return quantiles
        
class DQN_Policy(nn.Module):
    def __init__(self,
                 action_space,
                 representation,
                 activation,
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
        self.evalQ = BasicQhead(representation.output_shape['state'][0],self.action_dim,activation,initialize,device)
        self.target_representation = copy.deepcopy(self.eval_representation)
        self.targetQ = copy.deepcopy(self.evalQ)
    def forward(self,observation:dict):
        eval_outputs = self.eval_representation(observation)
        target_outputs = self.target_representation(observation)
        evalQ = self.evalQ(eval_outputs['state'])
        targetQ = self.targetQ(target_outputs['state'].detach()).detach()
        eval_outputs['evalQ'] = evalQ
        eval_outputs['targetQ'] = targetQ
        return eval_outputs,evalQ,targetQ

class DuelDQN_Policy(nn.Module):
    def __init__(self,
                 action_space,
                 representation,
                 activation,
                 initialize,
                 device):
        super(DuelDQN_Policy,self).__init__()
        assert isinstance(action_space,gym.spaces.Discrete), "DQN is not supported for non-discrete action space"
        self.action_dim = action_space.n
        self.representation = representation
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        self.output_shape['evalQ'] = (self.action_dim,)
        self.output_shape['targetQ'] = (self.action_dim,)
        
        self.eval_representation = representation
        self.evalQ = DuelQhead(representation.output_shape['state'][0],self.action_dim,activation,initialize,device)
        self.target_representation = copy.deepcopy(self.eval_representation)
        self.targetQ = copy.deepcopy(self.evalQ)
        
    def forward(self,observation:dict):
        eval_outputs = self.eval_representation(observation)
        target_outputs = self.target_representation(observation)
        evalQ = self.evalQ(eval_outputs['state'])
        targetQ = self.targetQ(target_outputs['state'])
        eval_outputs['evalQ'] = evalQ
        eval_outputs['targetQ'] = targetQ
        return eval_outputs,evalQ,targetQ.detach()

class C51DQN_Policy(nn.Module):
    def __init__(self,
                 action_space,
                 atom_num,
                 vmin,
                 vmax,
                 representation,
                 activation,
                 initialize,
                 device):
        super(C51DQN_Policy,self).__init__()
        assert isinstance(action_space,gym.spaces.Discrete), "DQN is not supported for non-discrete action space"
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        self.output_shape['evalQ_dist'] = (self.action_dim,self.atom_num,)
        self.output_shape['targetQ_dist'] = (self.action_dim,self.atom_num,)
        
        self.eval_representation = representation
        self.evalQ = C51Qhead(representation.output_shape['state'][0],self.action_dim,self.atom_num,activation,initialize,device)
        self.target_representation = copy.deepcopy(representation)
        self.targetQ = copy.deepcopy(self.evalQ)
        self.vmin = vmin
        self.vmax = vmax
        self.supports = nn.Parameter(torch.linspace(vmin,vmax,atom_num,device=device),requires_grad=False)
        self.delta = (vmax-vmin)/(atom_num-1)
    def forward(self,observation:dict):
        eval_outputs = self.eval_representation(observation)
        target_outputs = self.target_representation(observation)
        evalQ_dist = self.evalQ(eval_outputs['state'])
        targetQ_dist = self.targetQ(target_outputs['state'])
        eval_outputs['evalQ_dist'] = evalQ_dist
        eval_outputs['targetQ_dist'] = targetQ_dist
        return eval_outputs,evalQ_dist,targetQ_dist

class QRDQN_Policy(nn.Module):
    def __init__(self,
                 action_space,
                 quantile_num,
                 representation,
                 activation,
                 initialize,
                 device):
        super(QRDQN_Policy,self).__init__()
        assert isinstance(action_space,gym.spaces.Discrete), "DQN is not supported for non-discrete action space"
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        self.output_shape['evalQ_quantiles'] = (self.action_dim,self.quantile_num,)
        self.output_shape['targetQ_quantiles'] = (self.action_dim,self.quantile_num,)
        self.eval_representation = representation
        self.evalQ = QRDQNhead(representation.output_shape['state'][0],self.action_dim,self.quantile_num,activation,initialize,device)
        self.target_representation = copy.deepcopy(self.eval_representation)
        self.targetQ = copy.deepcopy(self.evalQ)
    def forward(self,observation:dict):
        eval_outputs = self.eval_representation(observation)
        target_outputs = self.target_representation(observation)
        evalQ_quantiles = self.evalQ(eval_outputs['state'])
        targetQ_quantiles = self.targetQ(target_outputs['state'])
        eval_outputs['evalQ_quantiles'] = evalQ_quantiles
        eval_outputs['targetQ_quantiles'] = targetQ_quantiles
        return eval_outputs,evalQ_quantiles,targetQ_quantiles


    