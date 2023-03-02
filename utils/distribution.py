from abc import abstractmethod
import torch
from torch.distributions import Categorical,Normal

class Distribution:
    def __init__(self):
        self.params_shape = None
        self.params = {}
        self.distribution = None
    def set_param(self,**kwargs):
        for key,value in zip(kwargs.keys(),kwargs.values()):
            self.params[key] = value
    def get_param(self):
        return self.params
    def get_distribution(self):
        return self.distribution
    def logprob(self,x:torch.Tensor):
        return self.distribution.log_prob(x)
    def entropy(self):
        return self.distribution.entropy()
    def sample(self):
        return self.distribution.sample()
    def kl_divergence(self,other_pd):
        return torch.distributions.kl_divergence(self.distribution,other_pd.distribution)

class CategoricalDistribution(Distribution):
    def __init__(self,action_dim:int):
        super().__init__()
        self.params_shape = {'logits':(action_dim,)}
    def set_param(self,**kwargs):
        super().set_param(**kwargs)
        self.distribution = Categorical(logits=self.params['logits'])
        
class DiagGaussianDistribution(Distribution):
    def __init__(self,action_dim:int):
        super().__init__()
        self.params_shape = {'mu':(action_dim,),'std':(action_dim,)}
    def set_param(self, **kwargs):
        super().set_param(**kwargs)
        self.distribution = Normal(self.params['mu'],self.params['std'])
    def logprob(self,x):
        return super().logprob(x).sum(-1)
    def entropy(self):
        return super().entropy().sum(-1)

class MultiheadDiagGaussianDistribution(Distribution):
    def __init__(self,action_dim:int,num_head:int):
        super().__init__()
        self.params_shape = {'mu':(num_head,action_dim,),'std':(num_head,action_dim,)}
    def set_param(self, **kwargs):
        super().set_param(**kwargs)
        self.distribution = Normal(self.params['mu'],self.params['std'])
    def logprob(self,x):
        return super().logprob(x).sum(-1)
    def entropy(self):
        return super().entropy().sum(-1)
    

    
    
        
        
        
        