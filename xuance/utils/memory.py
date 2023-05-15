from venv import create
import numpy as np
from .common import discount_cumsum
from typing import Optional, Union, Sequence

def create_memory(shape: Optional[Union[tuple, dict]], nenvs: int, nsize: int):
    if shape == None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in zip(shape.keys(), shape.values()):
            if value is None:  # save an object type
                if nenvs == 0:
                    memory[key] = np.zeros([nsize], dtype=object)
                else:
                    memory[key] = np.zeros([nenvs,nsize], dtype=object)
            else:
                if nenvs == 0:
                    memory[key] = np.zeros([nsize] + list(value), dtype=np.float32)
                else:
                    memory[key] = np.zeros([nenvs, nsize] + list(value), dtype=np.float32)
        return memory
    elif isinstance(shape, tuple):
        if nenvs == 0:
            return np.zeros([nsize] + list(shape), np.float32)
        else:
            return np.zeros([nenvs, nsize] + list(shape), np.float32)
    else:
        raise NotImplementedError

def store_element(data: Optional[Union[np.ndarray, dict, float]], memory: Union[dict, np.ndarray], ptr: int):
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in zip(data.keys(), data.values()):
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data

def store_batch_element(data: Optional[Union[np.ndarray, dict, float]], memory: Union[dict, np.ndarray], ptr: int):
    if data is None:
        return 
    elif isinstance(data,dict):
        for key,value in zip(data.keys(),data.values()):
            memory[key][ptr:ptr+value.shape[0]] = value
    else:
        memory[ptr:ptr+data.shape[0]] = data

def sample_batch(memory: Optional[Union[np.ndarray, dict]], index: np.ndarray):
    if memory is None:
        return None
    elif isinstance(memory, dict):
        batch = {}
        for key, value in zip(memory.keys(), memory.values()):
            batch[key] = value[index]
        return batch
    else:
        return memory[index]

class DummyOnPolicyBuffer:
    def __init__(self,
                 input_shape: dict,
                 action_shape: tuple,
                 output_shape: dict,
                 nenvs: int,
                 nsize: int,
                 nminibatch: int,
                 gamma: float=0.99,
                 tdlam: float=0.95):
        self.input_shape,self.action_shape,self.output_shape = input_shape,action_shape,output_shape
        self.size,self.ptr = 0,0
        self.nenvs,self.nsize,self.nminibatch = nenvs,nsize,nminibatch
        self.gamma,self.tdlam = gamma,tdlam
        self.start_ids = np.zeros(self.nenvs,np.int32)
        self.inputs = create_memory(input_shape,nenvs,nsize)
        self.actions = create_memory(action_shape,nenvs,nsize)
        self.outputs = create_memory(output_shape,nenvs,nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.returns = create_memory((),self.nenvs,self.nsize)
        self.advantages = create_memory((),self.nenvs,self.nsize)
    @property
    def full(self):
        return self.size >= self.nsize
    def clear(self):
        self.size,self.ptr = 0,0
        self.start_ids = np.zeros(self.nenvs,np.int32)
        self.inputs = create_memory(self.input_shape,self.nenvs,self.nsize)
        self.actions = create_memory(self.action_shape,self.nenvs,self.nsize)
        self.outputs = create_memory(self.output_shape,self.nenvs,self.nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.returns = create_memory((),self.nenvs,self.nsize)
        self.advantages = create_memory((),self.nenvs,self.nsize)
        
    def store(self,input,action,output,reward,value):
        store_element(input,self.inputs,self.ptr)
        store_element(action,self.actions,self.ptr)
        store_element(output,self.outputs,self.ptr)      
        store_element(reward,self.rewards,self.ptr)      
        store_element(value,self.returns,self.ptr)      
        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size+1,self.nsize)
    
    def finish_path(self, val, i):
        if self.full:
            path_slice = np.arange(self.start_ids[i], self.nsize).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
        rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
        critics = np.append(np.array(self.returns[i, path_slice]), [val], axis=0)
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        deltas = rewards[:-1] + self.gamma * critics[1:] - critics[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.tdlam)
        self.returns[i, path_slice] = returns
        self.advantages[i, path_slice] = advantages
        self.start_ids[i] = self.ptr
    
    def sample(self):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"
        env_choices = np.random.choice(self.nenvs, self.nenvs * self.nsize // self.nminibatch)
        step_choices = np.random.choice(self.nsize, self.nenvs * self.nsize // self.nminibatch)
        input_batch = sample_batch(self.inputs,tuple([env_choices, step_choices]))
        action_batch = sample_batch(self.actions,tuple([env_choices, step_choices]))
        output_batch = sample_batch(self.outputs,tuple([env_choices, step_choices]))
        return_batch = sample_batch(self.returns,tuple([env_choices, step_choices]))
        advantage_batch = sample_batch(self.advantages,tuple([env_choices, step_choices]))
        advantage_batch = (advantage_batch - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-7)
        return input_batch,action_batch,output_batch,return_batch,advantage_batch

class DummyOffPolicyBuffer:
    def __init__(self,
                 input_shape,
                 action_shape,
                 output_shape,
                 nenvs,
                 nsize,
                 minibatch,):
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.output_shape = output_shape
        self.nenvs,self.nsize,self.minibatch = nenvs,nsize,minibatch
        self.inputs = create_memory(input_shape,nenvs,nsize)
        self.actions = create_memory(action_shape,nenvs,nsize)
        self.outputs = create_memory(output_shape,nenvs,nsize)
        self.next_inputs = create_memory(input_shape,nenvs,nsize)
        self.rewards = create_memory((),nenvs,nsize)
        self.terminals = create_memory((),nenvs,nsize)
        self.ptr,self.size = 0,0
    def clear(self):
        self.ptr,self.size = 0,0
    def store(self,input,action,output,reward,terminal,next_input):
        store_element(input,self.inputs,self.ptr)
        store_element(action,self.actions,self.ptr)
        store_element(output,self.outputs,self.ptr)
        store_element(reward,self.rewards,self.ptr)
        store_element(terminal,self.terminals,self.ptr)
        store_element(next_input,self.next_inputs,self.ptr)
        self.ptr = (self.ptr+1)%self.nsize
        self.size = min(self.size+1,self.nsize)
    def sample(self):
        env_choices = np.random.choice(self.nenvs,self.minibatch)
        step_choices = np.random.choice(self.size,self.minibatch)
        input_batch = sample_batch(self.inputs,tuple([env_choices, step_choices]))
        action_batch = sample_batch(self.actions,tuple([env_choices, step_choices]))
        output_batch = sample_batch(self.outputs,tuple([env_choices, step_choices]))
        reward_batch = sample_batch(self.rewards,tuple([env_choices, step_choices]))
        terminal_batch = sample_batch(self.terminals,tuple([env_choices, step_choices]))
        next_input_batch = sample_batch(self.next_inputs,tuple([env_choices, step_choices]))        
        return input_batch,action_batch,output_batch,reward_batch,terminal_batch,next_input_batch
        

    
        
    
    
            
        
        
        
    
        
    
   
        
        
        
        
        
    

# class EpisodeOnPolicyBuffer:

# class DummyOffPolicyBuffer:



# class EpisodeOffPolicyBuffer: