from environment import *
class Running_MeanStd:
    def __init__(self,
                 shape:dict,
                 epsilon=1e-4):
        assert isinstance(shape,dict)
        self.shape = shape
        self.mean = {key:np.zeros(shape[key],np.float32) for key in shape.keys()}
        self.var = {key:np.ones(shape[key],np.float32) for key in shape.keys()}
        self.count = {key:epsilon for key in shape.keys()}
    
    @property
    def std(self):
        return {key:np.sqrt(self.var[key]) for key in self.shape.keys()}
    
    def update(self,x):
        batch_means = {}
        batch_vars = {}
        batch_counts = {}
        for key in self.shape.keys():
            if len(x[key].shape) == 1:
                batch_mean, batch_std, batch_count = np.mean(x[key][np.newaxis,:], axis=0), np.std(x[key][np.newaxis,:], axis=0), x[key][np.newaxis,:].shape[0]
            else:
                batch_mean, batch_std, batch_count = np.mean(x[key], axis=0), np.std(x[key], axis=0), x[key].shape[0]
            batch_means[key] = batch_mean
            batch_vars[key] = np.square(batch_std)
            batch_counts[key] = batch_count
        self.update_from_moments(batch_means, batch_vars, batch_counts) 
    
    def update_from_moments(self,batch_mean,batch_var,batch_count):
        for key in self.shape.keys():
            delta = batch_mean[key] - self.mean[key]
            tot_count = self.count[key] + batch_count[key]
            new_mean = self.mean[key] + delta * batch_count[key] / tot_count
            m_a = self.var[key] * (self.count[key])
            m_b = batch_var[key] * (batch_count[key])
            M2 = m_a + m_b + np.square(delta) * self.count[key] * batch_count[key] / (self.count[key] + batch_count[key])
            new_var = M2 / (self.count[key] + batch_count[key])
            new_count = batch_count[key] + self.count[key]
            self.mean[key] = new_mean
            self.var[key] = new_var
            self.count[key] = new_count
            

        
        
    
    
    
    
    
    
    
    