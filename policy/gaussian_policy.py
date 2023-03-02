from policy import *
class ActorNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 activation,
                 initialize,
                 device
                 ):
        super(ActorNet,self).__init__()
        self.device = device
        self.action_dim = action_dim
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),action_dim,None,initialize,device)
        self.model = nn.Sequential(*block1,*block2,nn.Tanh())
        self.logstd = nn.Parameter(-torch.ones((action_dim,),device=device))
        self.distribution = DiagGaussianDistribution(self.action_dim)
        self.output_shape = self.distribution.params_shape
    def forward(self,x:torch.Tensor):
        distribution = DiagGaussianDistribution(self.action_dim)
        distribution.set_param(mu=self.model(x),std=self.logstd.exp())
        return distribution.get_param(),distribution

class MultiheadActorNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 num_head:int,
                 activation,
                 initialize,
                 device):
        super(MultiheadActorNet,self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.mu_models = nn.ModuleList([nn.Sequential(*mlp_block(state_dim,action_dim,activation,initialize,device)[0]) for i in range(num_head)])
        self.logstds = nn.Parameter(-torch.ones((num_head,action_dim),device=device))
        self.distribution = DiagGaussianDistribution(self.action_dim)
        self.output_shape = self.distribution.params_shape
    def forward(self,x:torch.Tensor):
        mu = torch.concat([model(x) for model in self.mu_models])
        distribution = DiagGaussianDistribution(self.action_dim)
        distribution.set_param(mu,self.logstds.exp())
        return distribution.get_param(),distribution
                
class CriticNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 activation,
                 initialize,
                 device
                 ):
        super(CriticNet,self).__init__()
        self.device = device
        block1,_ = mlp_block(state_dim,max(state_dim,128),activation,initialize,device)
        block2,_ = mlp_block(max(state_dim,128),1,None,initialize,device)
        self.model = nn.Sequential(*block1,*block2)
        self.output_shape = {'critic':()}
    def forward(self,x:torch.Tensor):
        return self.model(x).squeeze(dim=-1)

# class MultiHeadCriticNet(nn.Module):
#     def __init__(self,
#                  state_dim:int,
#                  head_num:int,
#                  activation,
#                  initialize,
#                  device):
#         super(MultiHeadCriticNet,self).__init__()
#         self.device = device
#         self.model = nn.ModuleList([nn.Sequential(*mlp_block(state_dim,max(state_dim,128),activation,initialize,device)[0],
#                                     *mlp_block(max(state_dim,128),1,None,initialize,device)[0]) for i in range(head_num)])
#         self.output_shape = {'critic':(head_num,)}
#     def forward(self,x:torch.Tensor):
        

class ActorPolicy(nn.Module):
    def __init__(self,
                 action_space:gym.Space,
                 representation:ModuleType,
                 activation,
                 initialize,
                 device):
        assert isinstance(action_space, gym.spaces.Box)
        super(ActorPolicy,self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.input_shape = self.representation.input_shape.copy()
        self.output_shape = self.representation.output_shape.copy()
        self.output_shape['actor'] = self.action_space.shape
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.shape[0],
                              activation,
                              initialize,
                              device)
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
            
    def forward(self,observation:dict):
        outputs = self.representation(observation)
        a_param,a = self.actor(outputs['state'])
        for key in self.actor.output_shape.keys():
            outputs[key] = a_param[key]
        return outputs,a
    
class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space:gym.Space,
                 representation:ModuleType,
                 activation,
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
                              activation,
                              initialize,
                              device)
        self.critic = CriticNet(self.representation.output_shape['state'][0],
                                activation,
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

class PPGActorCritic(nn.Module):
    def __init__(self,
                 action_space:gym.Space,
                 representation:ModuleType,
                 activation,
                 initialize,
                 device):
        assert isinstance(action_space, gym.spaces.Discrete)
        super(PPGActorCritic,self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.critic_representation = copy.deepcopy(self.representation)
        
        self.input_shape = self.representation.input_shape.copy()
        self.output_shape = self.representation.output_shape.copy()
        
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.n,
                              activation,
                              initialize,
                              device)
        self.critic = CriticNet(self.representation.output_shape['state'][0],
                                activation,
                                initialize,
                                device)
        self.aux_critic = CriticNet(self.representation.output_shape['state'][0],
                                activation,
                                initialize,
                                device)
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
        self.output_shape['critic'] = ()
        self.output_shape['aux_critic'] = ()

    def forward(self,observation:dict):
        outputs = self.representation(observation)
        a_param,a = self.actor(outputs['state'])
        aux_v = self.aux_critic(outputs['state'])
        critic_states = self.critic_representation(observation)
        v = self.critic(critic_states['state'])
        for key in self.actor.output_shape.keys():
            outputs[key] = a_param[key]
        outputs['critic'] = v
        outputs['aux_critic'] = aux_v
        return outputs,a,v
    

