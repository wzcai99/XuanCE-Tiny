from xuance.policy import *
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
        self.mu = nn.Sequential(*mlp_block(state_dim,action_dim,None,initialize,device)[0],nn.Tanh())
        self.logstd = nn.Parameter(-torch.ones((action_dim,),device=device))
        self.distribution = DiagGaussianDistribution(self.action_dim)
        self.output_shape = self.distribution.params_shape
    def forward(self,x:torch.Tensor):
        distribution = DiagGaussianDistribution(self.action_dim)
        distribution.set_param(mu=self.mu(x),std=self.logstd.exp())
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
                 action_space:gym.Space,
                 representation:torch.nn.Module,
                 initialize,
                 device):
        assert isinstance(action_space, gym.spaces.Box)
        super(ActorCriticPolicy,self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.input_shape = self.representation.input_shape.copy()
        self.output_shape = self.representation.output_shape.copy()
        self.output_shape['actor'] = self.action_space.shape
        self.output_shape['critic'] = ()
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.shape[0],
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

class SACCriticNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 initialize,
                 device):
        super(CriticNet,self).__init__()
        self.device = device
        self.model = nn.Sequential(*mlp_block(state_dim+action_dim,state_dim,nn.LeakyReLU,initialize,device)[0],
                                   *mlp_block(state_dim,1,None,initialize,device)[0])
        self.output_shape = {'critic':()}
    def forward(self,x:torch.Tensor,a:torch.Tensor):
        return self.model(torch.concat((x,a),dim=-1))[:,0]
    
class SACPolicy(nn.Module):
    def __init__(self,
                 action_space:gym.Space,
                 representation:torch.nn.Module,
                 initialize,
                 device):
        assert isinstance(action_space, gym.spaces.Box)
        super(SACPolicy,self).__init__()
        self.action_space = action_space
        self.input_shape = self.representation.input_shape.copy()
        self.output_shape = self.representation.output_shape.copy()
        
        self.representation = representation
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.shape[0],
                              initialize,
                              device)
        
        self.criticA_representation = copy.deepcopy(representation)
        self.criticA = SACCriticNet(self.representation.output_shape['state'][0],
                                    self.action_space.shape[0],
                                    initialize,
                                    device)
        
        self.criticB_representation = copy.deepcopy(representation)
        self.criticB = SACCriticNet(self.representation.output_shape['state'][0],
                                    self.action_space.shape[0],
                                    initialize,
                                    device)
        
        self.targetA_critic_representation = copy.deepcopy(representation)
        self.targetB_critic_representation = copy.deepcopy(representation)
        self.target_criticA = copy.deepcopy(self.criticA)
        self.target_criticB = copy.deepcopy(self.criticB)
        
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
        self.output_shape['criticA'] = ()
        self.output_shape['criticB'] = ()
        self.actor_parameters = list(self.representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.criticA_representation.parameters()) + list(self.criticA.parameters()) + list(self.criticB_representation.parameters()) + list(self.criticB.parameters())
        