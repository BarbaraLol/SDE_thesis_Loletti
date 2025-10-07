import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embedding_size=256, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        if config.model.embedding_type == 'fourier':
            self.time_embed = GaussianFourierProjection(
                embedding_size=config.model.nf,
                scale=config.model.fourier_scale
            )
            embed_dim = 2 * config.model.nf
        else:
            self.time_embed = nn.Sequential(
                nn.Linear(1, config.model.nf),
                nn.SiLU(),
            )
            embed_dim = config.model.nf
            
        input_dim = 2
        hidden_dim = config.model.nf
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Residual blocks with time conditioning
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.model.dropout),
            ) for _ in range(config.model.num_res_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 2)
        
        # Initialize output to near-zero for stability
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Reshape from (batch, 2, 1, 1) to (batch, 2)
        if len(x.shape) == 4:
            x_flat = x.view(batch_size, 2)
        else:
            x_flat = x
            
        # Time embedding
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        elif len(t.shape) == 3:
            t = t.view(batch_size, -1)
            
        time_emb = self.time_embed(t)
        
        # Ensure time_emb is 2D
        if len(time_emb.shape) > 2:
            time_emb = time_emb.view(batch_size, -1)
        
        # Forward pass with residual connections
        h = self.input_layer(x_flat)
        
        for block in self.blocks:
            h_input = torch.cat([h, time_emb], dim=1)
            # h = h + block(h_input)  # Residual connection
            h = block(h_input)  # Without residual connection
            
            
        score = self.output_layer(h)
        
        # Reshape back to (batch, 2, 1, 1) if needed
        if len(x.shape) == 4:
            score = score.view(batch_size, 2, 1, 1)
            
        return score