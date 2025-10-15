"""
Denoising Score Matching for learning the score function
Uses sinusoidal time embeddings for better temporal modeling
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Callable
from tqdm import tqdm


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (Transformer/DDPM-style)
    Converts scalar time values into high-dimensional vectors
    """
    def __init__(self, embed_dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.freqs = None  # Created lazily on first forward pass
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: time values (batch_size,)
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        if self.freqs is None:
            # Create frequency values
            half = self.embed_dim // 2
            device = t.device
            freqs = torch.exp(
                -math.log(self.max_period) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
            )
            self.freqs = freqs
        
        # Compute angles: (batch, half)
        angles = t[:, None] * self.freqs[None, :]
        
        # Apply sin and cos
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # Pad if embed_dim is odd
        if emb.shape[1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[1]), value=0.0)
        
        return emb


class ScoreNet(nn.Module):
    """
    Neural network to learn the score function ∇log p_t(x)
    Uses sinusoidal time embeddings for improved temporal modeling
    """
    def __init__(self, dim: int, hidden_dim: int = 128, n_layers: int = 3, 
                 time_embedding_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.time_embedding_dim = time_embedding_dim
        
        # Sinusoidal time embedding
        self.time_embed = TimeEmbedding(embed_dim=time_embedding_dim)
        
        # Main network
        layers = []
        input_dim = dim + time_embedding_dim
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: positions (batch_size, dim)
            t: time (batch_size,) or (batch_size, 1)
        Returns:
            score: ∇log p_t(x) with shape (batch_size, dim)
        """
        if t.dim() == 1:
            t_flat = t
        else:
            t_flat = t.squeeze(-1)
        
        t_embed = self.time_embed(t_flat)
        h = torch.cat([x, t_embed], dim=-1)
        score = self.network(h)
        return score


class DenoisingScoreMatcher:
    """
    Trains a neural network to approximate the score function using denoising
    """
    def __init__(self,
                 forward_trajectory: List[np.ndarray],
                 forward_times: np.ndarray,
                 dim: int,
                 device: str = 'cpu',
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 time_embedding_dim: int = 32):
        
        self.trajectory = forward_trajectory
        self.times = forward_times
        self.dim = dim
        self.device = device
        
        # Initialize neural network with sinusoidal time embeddings
        self.score_net = ScoreNet(
            dim=dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_embedding_dim=time_embedding_dim
        ).to(device)
        
        n_params = sum(p.numel() for p in self.score_net.parameters())
        print(f"✓ ScoreNet initialized ({n_params} parameters)")
    
    def train(self,
              n_epochs: int = 1000,
              batch_size: int = 64,
              lr: float = 1e-3,
              sigma_dn: float = 0.1,
              weight_decay: float = 0.0,
              verbose: bool = True) -> List[float]:
        """
        Train the score network using denoising score matching
        
        Loss: E[||s_θ(x̃,t) + (1/σ²_dn)(x̃-x)||²]
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            sigma_dn: Denoising noise level
            weight_decay: L2 regularization
            verbose: Print progress
            
        Returns:
            loss_history: List of average losses per epoch
        """
        optimizer = optim.Adam(self.score_net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_history = []
        
        # Create dataset: (x, t) pairs from all trajectory snapshots
        dataset = []
        for positions, time in zip(self.trajectory, self.times):
            for pos in positions:
                dataset.append((pos, time))
        
        n_data = len(dataset)
        
        self.score_net.train()
        pbar = tqdm(range(n_epochs), disable=not verbose, desc="Training")
        
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle dataset
            indices = np.random.permutation(n_data)
            
            for i in range(0, n_data, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Sample x ~ p̂_t (from dataset)
                x_batch = np.array([dataset[idx][0] for idx in batch_indices])
                t_batch = np.array([dataset[idx][1] for idx in batch_indices])
                
                x_clean = torch.FloatTensor(x_batch).to(self.device)
                t = torch.FloatTensor(t_batch).to(self.device)
                
                # Sample ε ~ N(0, I)
                epsilon = torch.randn_like(x_clean)
                
                # Corrupt: x̃ = x + σ_dn * ε
                x_noisy = x_clean + sigma_dn * epsilon
                
                # Predict score at noisy point
                score_pred = self.score_net(x_noisy, t)
                
                # Target: -ε/σ_dn = -(1/σ²_dn)(x̃-x)
                target = -epsilon / sigma_dn
                
                # Denoising score matching loss
                loss = torch.mean((score_pred - target) ** 2)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            
            if verbose:
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # Print periodic updates
            if (epoch + 1) % 100 == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")
        
        self.score_net.eval()
        print(f"\n✓ Training completed! Final loss: {loss_history[-1]:.6f}")
        return loss_history
    
    def get_score_function(self) -> Callable:
        """
        Returns a callable score function for backward SDE
        
        Returns:
            score_fn: Function (x: np.ndarray, t: float) -> np.ndarray
        """
        self.score_net.eval()
        
        def score_fn(x: np.ndarray, t: float) -> np.ndarray:
            """
            Evaluate the learned score function
            
            Args:
                x: positions (n_points, dim)
                t: time (scalar)
            Returns:
                score: ∇log p_t(x) with shape (n_points, dim)
            """
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                t_tensor = torch.FloatTensor([t] * x.shape[0]).to(self.device)
                score_tensor = self.score_net(x_tensor, t_tensor)
                return score_tensor.cpu().numpy()
        
        return score_fn
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.score_net.state_dict(),
            'dim': self.dim,
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.score_net.load_state_dict(checkpoint['model_state_dict'])
        self.score_net.eval()
        print(f"✓ Model loaded from {path}")