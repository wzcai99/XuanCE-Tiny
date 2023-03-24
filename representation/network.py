from representation import *
class MLP(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_sizes,
                 activation,
                 initialize,
                 device):
        super(MLP,self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.output_shape = {'state':(hidden_sizes[-1],)}
        layers = []
        input_shape = self.input_shape['observation']
        for h in hidden_sizes:
            block,input_shape = mlp_block(input_shape[0],h,activation,initialize,device)
            layers.extend(block)
        self.model = nn.Sequential(*layers)
    def forward(self,observation: dict):
        tensor_observation = torch.as_tensor(observation['observation'],dtype=torch.float32,device=self.device)
        state = self.model(tensor_observation)
        return {'state':state}

class CNN(nn.Module):
    def __init__(self,
                 input_shape,
                 filters,
                 kernels,
                 strides,
                 activation,
                 initialize,
                 device):
        super(CNN,self).__init__()
        self.device = device
        self.input_shape = input_shape
        layers = []
        input_shape = self.input_shape['observation']
        for f,k,s in zip(filters,kernels,strides):
            block,input_shape = cnn_block(input_shape,f,k,s,activation,initialize,device)
            layers.extend(block)
        layers.append(nn.Flatten())
        self.output_shape = {'state':(np.prod(layers),)}
        self.model = nn.Sequential(*layers)
        
    def forward(self,observation: dict):
        tensor_observation = torch.as_tensor(observation['observation']/255.0,dtype=torch.float32,device=self.device).permute(0,3,1,2)
        state = self.model(tensor_observation)
        return {'state':state}



