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
        self.model = nn.Sequential(*mlp_block(state_dim,action_dim,nn.Tanh,initialize,device)[0])
        self.output_shape = {'actor':(action_dim,)}
    def forward(self,x:torch.Tensor):
        return self.model(x)
class CriticNet(nn.Module):
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
      
class DDPGPolicy(nn.Module):
    def __init__(self,
                 action_space:gym.spaces.Space,
                 representation:torch.nn.Module,
                 initialize,
                 device):
        assert isinstance(action_space,gym.spaces.Box)
        super(DDPGPolicy,self).__init__()
        self.action_space = action_space
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        # dont share the representation network in actor and critic
        self.representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.target_actor_representation = copy.deepcopy(representation)
        self.target_critic_representation = copy.deepcopy(representation)
        
        # create actor,critic and target actor, target critic
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.shape[0],
                              initialize,
                              device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = CriticNet(self.representation.output_shape['state'][0],
                                self.action_space.shape[0],
                                initialize,
                                device)
        self.target_critic = copy.deepcopy(self.critic)
        
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
        self.output_shape['critic'] = ()
        self.actor_parameters = list(self.representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_representation.parameters()) + list(self.critic.parameters())
    
    def forward(self,observation:dict):
        actor_outputs = self.representation(observation)
        critic_outputs = self.critic_representation(observation)
        action = self.actor(actor_outputs['state'])
        critic = self.critic(critic_outputs['state'],action)
        actor_outputs['actor'] = action
        actor_outputs['critic'] = critic
        return actor_outputs,action,critic
    
    def Qtarget(self,observation:dict):
        actor_outputs = self.target_actor_representation(observation)
        critic_outputs = self.target_critic_representation(observation)
        target_action = self.target_actor(actor_outputs['state']).detach()
        target_critic = self.target_critic(critic_outputs['state'],target_action)
        return target_critic.detach()
    
    def Qaction(self,observation:dict,action:torch.Tensor):
        outputs = self.critic_representation(observation)
        critic = self.critic(outputs['state'],action)
        return critic
    
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)

class TD3Policy(nn.Module):
    def __init__(self,
                 action_space:gym.spaces.Space,
                 representation:torch.nn.Module,
                 initialize,
                 device):
        assert isinstance(action_space,gym.spaces.Box)
        super(TD3Policy,self).__init__()
        self.action_space = action_space
        self.input_shape = representation.input_shape.copy()
        self.output_shape = representation.output_shape.copy()
        # dont share the representation network in actor and critic
        
        self.representation = representation
        self.criticA_representation = copy.deepcopy(representation)
        self.criticB_representation = copy.deepcopy(representation)
        
        self.target_actor_representation = copy.deepcopy(representation)
        self.targetA_critic_representation = copy.deepcopy(representation)
        self.targetB_critic_representation = copy.deepcopy(representation)
        
        # create actor,critic and target actor, target critic
        self.actor = ActorNet(self.representation.output_shape['state'][0],
                              self.action_space.shape[0],
                              initialize,
                              device)
        self.target_actor = copy.deepcopy(self.actor)
       
        self.criticA = CriticNet(self.representation.output_shape['state'][0],
                                self.action_space.shape[0],
                                initialize,
                                device)
        self.criticB = CriticNet(self.representation.output_shape['state'][0],
                                self.action_space.shape[0],
                                initialize,
                                device)
        self.target_criticA = copy.deepcopy(self.criticA)
        self.target_criticB = copy.deepcopy(self.criticB)
        
        for key,value in zip(self.actor.output_shape.keys(),self.actor.output_shape.values()):
            self.output_shape[key] = value
        self.output_shape['criticA'] = ()
        self.output_shape['criticB'] = ()
        self.actor_parameters = list(self.representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.criticA_representation.parameters()) + list(self.criticA.parameters()) + list(self.criticB_representation.parameters()) + list(self.criticB.parameters())
    
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticA_representation.parameters(), self.targetA_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticB_representation.parameters(), self.targetB_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticA.parameters(), self.target_criticA.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticB.parameters(), self.target_criticB.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
    
    def Qtarget(self,observation:dict):
        actor_outputs = self.target_actor_representation(observation)
        criticA_outputs = self.targetA_critic_representation(observation)
        criticB_outputs = self.targetB_critic_representation(observation)
        target_action = self.target_actor(actor_outputs['state']).detach()
        target_action = target_action + torch.rand_like(target_action).clamp(-0.1,0.1)
        targetA_critic = self.target_criticA(criticA_outputs['state'],target_action)
        targetB_critic = self.target_criticB(criticB_outputs['state'],target_action)
        return torch.minimum(targetA_critic,targetB_critic).detach()
    
    def forward(self,observation:dict):
        actor_outputs = self.representation(observation)
        criticA_outputs = self.criticA_representation(observation)
        criticB_outputs = self.criticB_representation(observation)
        action = self.actor(actor_outputs['state'])
        criticA = self.criticA(criticA_outputs['state'],action)
        criticB = self.criticB(criticB_outputs['state'],action)
        actor_outputs['actor'] = action
        actor_outputs['criticA'] = criticA
        actor_outputs['criticB'] = criticB
        return actor_outputs,action,(criticA+criticB)/2

    def Qaction(self,observation:dict,action:torch.Tensor):
        outputsA = self.criticA_representation(observation)
        criticA = self.criticA(outputsA['state'],action)
        outputsB = self.criticB_representation(observation)
        criticB = self.criticB(outputsB['state'],action)
        return criticA,criticB




