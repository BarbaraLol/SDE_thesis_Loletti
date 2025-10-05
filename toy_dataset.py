import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Toy2DDataset(Dataset):
    """2D toy dataset with minimal spatial dimensions."""
    
    def __init__(self, n_samples=10000, toy_type='gaussian'):
        self.n_samples = n_samples
        
        # Generate 2D data points
        if toy_type == 'gaussian':
            data = np.random.randn(n_samples, 2) * 2.0
        elif toy_type == 'circle':
            angles = np.random.rand(n_samples) * 2 * np.pi
            radius = 2.0
            data = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        elif toy_type == 'spiral':
            t = np.linspace(0, 4*np.pi, n_samples)
            r = t / (2*np.pi)
            data = np.column_stack([r * np.cos(t), r * np.sin(t)])
        else:
            data = np.random.randn(n_samples, 2) * 2.0
        
        # Convert to format with minimal spatial dimensions: (N, 2, 1, 1)
        # This is honest about the data structure while maintaining CNN compatibility
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data_point = self.data[idx].unsqueeze(-1).unsqueeze(-1)  # (2,) -> (2, 1, 1)
        return {'image': data_point}


def get_data_loaders(config):
    """Create train and test data loaders."""
    train_dataset = Toy2DDataset(
        n_samples=8000, 
        toy_type=getattr(config.data, 'toy_type', 'gaussian')
    )
    test_dataset = Toy2DDataset(
        n_samples=2000, 
        toy_type=getattr(config.data, 'toy_type', 'gaussian')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    return train_loader, test_loader, test_loader


def dataloader_to_iterator(dataloader):
    """Convert PyTorch DataLoader to iterator format expected by run_lib.py"""
    while True:
        for batch in dataloader:
            # Convert to format expected by the training loop
            yield batch