"""Dataset generator for quasipotential analysis"""

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
        self.dt = config.data.dt
        
        # Initialize dynamical system
        if config.data.example == 'example1':
            self.system = Example1System()
        elif config.data.example == 'example2':
            self.system = Example2System()
        else:
            raise ValueError(f"Unknown example: {config.data.example}")
        
        # Generate trajectory data
        self.data_pairs = self._generate_data()
        
    def _generate_data(self):
        """Generate (x_t, x_{t+dt}) pairs from trajectories"""
        n_traj = self.config.data.n_trajectories
        T = self.config.data.T
        dt = self.dt
        domain = self.config.data.domain
        
        all_pairs = []
        
        for _ in range(n_traj):
            # Random initial condition
            x0 = np.array([np.random.uniform(d[0], d[1]) for d in domain])
            
            # Integrate ODE
            t_eval = np.arange(0, T, dt)
            traj = odeint(lambda x, t: self.system.f(x), x0, t_eval)
            
            # Create (x_t, x_{t+dt}) pairs
            for i in range(len(traj) - 1):
                all_pairs.append((traj[i], traj[i+1]))
        
        # Sample subset
        n_samples = min(20000, len(all_pairs))
        indices = np.random.choice(len(all_pairs), n_samples, replace=False)
        sampled_pairs = [all_pairs[i] for i in indices]
        
        return sampled_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        x, x_next = self.data_pairs[idx]

        # Convert to tensors and add spatial dims
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

        return {
            'x': x_tensor,  # (dim, 1, 1)
            'x_next': x_next_tensor,  # (dim, 1, 1)
            'dt': torch.tensor(self.dt, dtype=torch.float32)  # Make it a tensor scalar
        }


def get_quasipotential_dataset(config, evaluation=False):
    """Create data loaders"""
    dataset = QuasipotentialDataset(config, evaluation)
    
    batch_size = config.eval.batch_size if evaluation else config.training.batch_size
    
    # Split into train/test
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    return train_loader, test_loader, dataset.system