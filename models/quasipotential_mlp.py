''' 
    NN implementation for learning the quasipotential via orthogonal decomposition (as in the paper)
'''

import torch
import torch.nn as nn
from models import utils as mutils

@mutils.register_model(name='quasipotential_mlp')
class GeneralizedQuasipotential(nn.Module):
    ''' 
        Learn V(x) and g(x) such that f(x) = -∇V(x) + g(x)
        Considering the orthogonal constraint ∇V(x)^Tg(x) = 0
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dims = config.model.dim
        hidden_dims = config.model.hidden_dims
        activation = config.model.activation

        # Network for the V(x) - scalar potential
        v_layers = []
        prev_dim = self.dims
        for h_dim in hidden_dims:
            v_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = h_dim
        v_layers.append(nn.Linear(prev_dim, 1))
        self.v_net = nn.Sequential(*v_layers)

        # Network for g(x) - rotational component
        g_layers = []
        prev_dim = self.dims
        for h_dim in hidden_dims:
            g_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = h_dim
        g_layers.append(nn.Linear(prev_dim, self.dims))
        self.g_net = nn.Sequential(*g_layers)

        # Quadratic term 
        self.quadratic_weight = nn.Parameter(torch.ones(1) * 0.1)

    
    def compute_V(self, x):
        '''V(x) = V_net(x) + quadratic term'''
        if len(x.shape) == 4:
            x = x.squeeze(-1).squeeze(-1)
        
        v_net = self.v_net(x)
        quadratic = 0.5 * self.quadratic_weight * torch.sum(x**2, dim=-1, keepdim=True)
        return v_net + quadratic        

    def compute_grad_V(self, x):
        '''Using automatic differentiation to compute the gradient'''
        if len(x.shape) == 4:
            x = x.squeeze(-1).squeeze(-1)
        
        # Ensure x requires grad
        if not x.requires_grad:
            x = x.requires_grad_(True)
        # x_req_grad = x.requires_grad_(True)
        
        v = self.compute_V(x_req_grad)
        grad_v = torch.autograd.grad(v.sum(), x_req_grad, create_graph=True)[0]
        return grad_v
    
    def compute_g(self, x):
        '''compute g(x), the rotational component'''
        if len(x.shape) == 4:
            x = x.squeeze(-1).squeeze(-1)
        return self.g_net(x)

    def forward(self, x, t=None):
        ''' 
        It returns the constructed vector field f = -∇V(x) + g(x)
        '''
        # Ensure x requires grad for autograd
        if len(x.shape) == 4:
            x_flat = x.squeeze(-1).squeeze(-1)
        else:
            x_flat = x
        
        if not x_flat.requires_grad:
            x_flat = x_flat.requires_grad_(True)
        
        grad_v = self.compute_grad_V(x_flat)
        g = self.compute_g(x_flat)
        f = -grad_v + g

        # Return in expected format
        if len(x.shape) == 4:
            return f.unsqueeze(-1).unsqueeze(-1)
        return f


def create_model(config):
    """Factory function to create the model"""
    return GeneralizedQuasipotential(config)