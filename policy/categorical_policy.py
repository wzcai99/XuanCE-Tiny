from policy import *
class ActorNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 initialize,
                 device
                 ):
        super(ActorNet,self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.model = nn.Sequential(*mlp_block(state_dim,action_dim,None,initialize,device)[0])
        self.distribution = CategoricalDistribution(action_dim)
        self.output_shape = self.distribution.params_shape
    def forward(self,x:torch.Tensor):
        distribution = CategoricalDistribution(self.action_dim)
        distribution.set_param(logits=self.model(x))
        return distribution.get_param(),distribution

class CriticNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 initialize,
                 device
                 ):
        super(CriticNet,self).__init__()
        self.device = device
        self.model = nn.Sequential(*mlp_block(state_dim,1,None,initialize,device)[0])
        self.output_shape = {'critic':()}
    def forward(self,x:torch.Tensor):
        return self.model(x).squeeze(dim=-1)
    
class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space:gym.spaces.Discrete,
                 representation:torch.nn.Module,
                 initialize,
                 device):
        assert isinstance(action_space, gym.spaces.Discrete)
        super(ActorCriticPolicy,self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.input_shape = self.representation.input_shape.copy()
        self.output_shape = self.representation.output_shape.copy()
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.n,
                              initialize,
                              device)
        self.critic = CriticNet(self.representation.output_shape['state'][0],
                                initialize,
                                device)
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
        self.output_shape['critic'] = ()

    def forward(self,observation:dict):
        outputs = self.representation(observation)
        a_param,a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        for key in self.actor.output_shape.keys():
            outputs[key] = a_param[key]
        outputs['critic'] = v
        return outputs,a,v
        