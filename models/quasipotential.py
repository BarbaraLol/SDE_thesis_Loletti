''' 
    NN implementation for learning the quasipotential via orthogonal decomposition (as in the paper)
'''

import torch
import torch.nn as nn

class GeneralizedQuasipotential(Module.nn):
    ''' 
        Learn V(x) and g(x) such that f(x) = -∇V(x) + g(x)
        Considering the orthogonal constraint ∇V(x)^Tg(x) = 0
    '''
    def __init__(self, config):
        super.__init__()
        self.config = config

    
    def compute_V(self, x):
        '''V(x) = ...............'''

    def compute_grad_V(self, x):
        ''''''
    
    def compute_g(self, x):
        '''compute the g(x) component'''

    def forward(self, x):
        ''' 
        It returns
            - f (constructed vector field)
            - grad_v (gradient of V)
            - g (rotational component)
        '''
        grad_v = self.compute_grad_V(x)
        g = self.compute_g(x)
        f = -grad_v + g

        return f, grad_v, g 
