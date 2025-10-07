'''Dataset generator for the quasipotential analysis'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint
from dynamical_quasipotential_system import Example1System, Example2System

class QuasipotentialDataset(Dataset):
    """Dataset from dynamical system trajectories"""
    
    def __init__(self, config, evaluation=False):
        self.config = config
        self.dim = config.data.dim
        
        # Initialize dynamical system
        if config.data.example == 'example1':
            self.system = Example1System()
        elif config.data.example == 'example2':
            self.system = Example2System()
        else:
            raise ValueError(f"Unknown example: {config.data.example}")
        
        # Generate trajectories
        self.data_points = self._generate_data()
        
    def _generate_data(self):
        """Generate training data from trajectories"""
        n_traj = self.config.data.n_trajectories
        T = self.config.data.T
        dt = self.config.data.dt
        domain = self.config.data.domain
        
        all_points = []
        
        for _ in range(n_traj):
            # Random initial condition
            x0 = np.array([np.random.uniform(d[0], d[1]) for d in domain])
            
            # Integrate ODE
            t = np.arange(0, T, dt)
            traj = odeint(lambda x, t: self.system.f(x), x0, t)
            all_points.append(traj)
        
        # Combine all points
        all_points = np.vstack(all_points)
        
        # Sample subset to avoid memory issues
        n_samples = min(20000, len(all_points))
        indices = np.random.choice(len(all_points), n_samples, replace=False)
        
        return torch.tensor(all_points[indices], dtype=torch.float32)
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        x = self.data_points[idx]
        # Format: (dim, 1, 1) for compatibility
        x = x.unsqueeze(-1).unsqueeze(-1)
        return {'image': x, 'vector_field': self.system.f(self.data_points[idx])}

def get_quasipotential_dataset(config, evaluation=False):
    """Create data loaders"""
    dataset = QuasipotentialDataset(config, evaluation)
    
    batch_size = config.eval.batch_size if evaluation else config.training.batch_size
    
    # Split into train/test
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader, dataset.system