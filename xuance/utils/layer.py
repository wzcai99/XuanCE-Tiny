import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union, Callable
ModuleType = Type[nn.Module]

class NoisyLinear(nn.Linear):
    def __init__(self,
                 input_features:int,
                 output_features:int,
                 sigma: float = 0.02,
                 bias: bool = True,
                 dtype: Any = None,
                 device: Any = None):
        super().__init__(input_features,output_features,bias,device,dtype)
        sigma_init = sigma / np.sqrt(input_features)
        self.sigma_weight = nn.Parameter(torch.ones((output_features,input_features),dtype=dtype,device=device)*sigma_init)
        self.register_buffer("epsilon_input", torch.zeros(1, input_features,dtype=dtype,device=device))
        self.register_buffer("epsilon_output", torch.zeros(output_features, 1,dtype=dtype,device=device))
        if bias:
            self.sigma_bias = nn.Parameter(torch.ones((output_features,),dtype=dtype,device=device)*sigma_init)
            
    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias) 
    