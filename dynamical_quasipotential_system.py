"""Dynamical systems for Examples 1 and 2"""
# Aiuto da CLAUDE e ancora devo verificare che sia corretto

import numpy as np
import torch
from scipy.integrate import odeint

class Example1System:
    """3D system with two stable equilibria (Eq. 19)"""
    def __init__(self):
        self.dim = 3
        self.name = "Example1"
    
    def f(self, x):
        """Vector field"""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x_val, y_val, z_val = x[0], x[1], x[2]
                dx = -2*(x_val**3 - x_val) - (y_val + z_val)
                dy = -y_val + 2*(x_val**3 - x_val)
                dz = -z_val + 2*(x_val**3 - x_val)
                return np.array([dx, dy, dz])
            else:
                x_val, y_val, z_val = x[:, 0], x[:, 1], x[:, 2]
                dx = -2*(x_val**3 - x_val) - (y_val + z_val)
                dy = -y_val + 2*(x_val**3 - x_val)
                dz = -z_val + 2*(x_val**3 - x_val)
                return np.stack([dx, dy, dz], axis=1)
        else:
            x_val, y_val, z_val = x[:, 0], x[:, 1], x[:, 2]
            dx = -2*(x_val**3 - x_val) - (y_val + z_val)
            dy = -y_val + 2*(x_val**3 - x_val)
            dz = -z_val + 2*(x_val**3 - x_val)
            return torch.stack([dx, dy, dz], dim=1)
    
    def true_quasipotential(self, x):
        """U(x,y,z) = (1-x²)² + y² + z²"""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return (1 - x[0]**2)**2 + x[1]**2 + x[2]**2
            return (1 - x[:, 0]**2)**2 + x[:, 1]**2 + x[:, 2]**2
        return (1 - x[:, 0]**2)**2 + x[:, 1]**2 + x[:, 2]**2


class Example2System:
    """2D system with limit cycle (Eq. 22-24)"""
    def __init__(self, a=1.0, b=2.5):
        self.dim = 2
        self.a = a
        self.b = b
        self.name = "Example2"
    
    def U(self, x):
        """Quasipotential"""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x_val, y_val = x[0], x[1]
            else:
                x_val, y_val = x[:, 0], x[:, 1]
        else:
            x_val, y_val = x[:, 0], x[:, 1]
        
        term = (x_val - self.a)**2 + (x_val - self.a)*(y_val - self.b) + (y_val - self.b)**2 - 0.5
        return term**2
    
    def grad_U(self, x):
        """∇U"""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x_val, y_val = x[0], x[1]
            else:
                x_val, y_val = x[:, 0], x[:, 1]
            inner = (x_val - self.a)**2 + (x_val - self.a)*(y_val - self.b) + (y_val - self.b)**2 - 0.5
            dU_dx = 2 * inner * (2*(x_val - self.a) + (y_val - self.b))
            dU_dy = 2 * inner * ((x_val - self.a) + 2*(y_val - self.b))
            if x.ndim == 1:
                return np.array([dU_dx, dU_dy])
            return np.stack([dU_dx, dU_dy], axis=1)
        else:
            x_val, y_val = x[:, 0], x[:, 1]
            inner = (x_val - self.a)**2 + (x_val - self.a)*(y_val - self.b) + (y_val - self.b)**2 - 0.5
            dU_dx = 2 * inner * (2*(x_val - self.a) + (y_val - self.b))
            dU_dy = 2 * inner * ((x_val - self.a) + 2*(y_val - self.b))
            return torch.stack([dU_dx, dU_dy], dim=1)
    
    def f(self, x):
        """Vector field"""
        grad_u = self.grad_U(x)
        
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x_val, y_val = x[0], x[1]
                rot_x = -2*(x_val + 2*y_val - self.a - 2*self.b)
                rot_y = 2*(2*x_val + y_val - 2*self.a - self.b)
                rot = np.array([rot_x, rot_y])
            else:
                x_val, y_val = x[:, 0], x[:, 1]
                rot_x = -2*(x_val + 2*y_val - self.a - 2*self.b)
                rot_y = 2*(2*x_val + y_val - 2*self.a - self.b)
                rot = np.stack([rot_x, rot_y], axis=1)
            return -0.5 * grad_u + rot
        else:
            x_val, y_val = x[:, 0], x[:, 1]
            rot_x = -2*(x_val + 2*y_val - self.a - 2*self.b)
            rot_y = 2*(2*x_val + y_val - 2*self.a - self.b)
            rot = torch.stack([rot_x, rot_y], dim=1)
            return -0.5 * grad_u + rot
    
    def true_quasipotential(self, x):
        """Known quasipotential"""
        return self.U(x)