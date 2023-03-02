from representation import *
from utils.layer import mlp_block

class Identical(nn.Module):
    def __init__(self,input_shape,device):
        super(Identical,self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.output_shape = {'state':input_shape['observation']}
        self.model = None
    def forward(self,observation:dict):
        state = torch.as_tensor(observation['observation'],dtype=torch.float32,device=self.device)
        return {'state':state}
    
class MLP(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_sizes,
                 initialize,
                 activation,
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
    
class MLP_MT(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_sizes,
                 initialize,
                 activation,
                 device):
        super(MLP_MT,self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.output_shape = {'state':(hidden_sizes[-1]*2,)}
        layers = []
        input_shape = self.input_shape['observation']
        for h in hidden_sizes:
            block,input_shape = mlp_block(input_shape[0],h,activation,initialize,device)
            layers.extend(block)
        self.model = nn.Sequential(*layers)
        self.one_hot_embedding = nn.Linear(10,hidden_sizes[-1],device=device)
    def forward(self,observation: dict):
        tensor_observation = torch.as_tensor(observation['observation'],dtype=torch.float32,device=self.device)
        tensor_onehot = torch.as_tensor(observation['task_index'],dtype=torch.float32,device=self.device).long()
        state = self.model(tensor_observation)
        one_hot_state = self.one_hot_embedding(F.one_hot(tensor_onehot,10).float()[:,0])
        return {'state':torch.concat((state,one_hot_state),dim=-1)}


