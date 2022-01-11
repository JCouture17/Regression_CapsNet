## Misc. Functions ##
'''
These functions are used to load the data files. 
'''

import numpy as np
import pandas as pd
from scipy import io

class Functions:
    def extract(self, filename, variable_name): 
        C = io.loadmat(filename, mat_dtype=True)
        C = C[variable_name] 
        C = np.squeeze(C)
        self.dt = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
        for i in range(self.dt.shape[0]):
            self.dt[i] = C[i]
        return self.dt
    
    def load(self, filename, variable_name): # Load from a Matlab (.mat) file
        C = io.loadmat(filename)
        self.C = C[variable_name] # Requires the name of the variable under which it was saved within Matlab
        return self.C
    
    def read(self, filename):
        self.dt = pd.read_excel(filename)
        return self.dt
    
    def reshape(self, variable, size):
        x = variable.reshape(size)
        return x 

