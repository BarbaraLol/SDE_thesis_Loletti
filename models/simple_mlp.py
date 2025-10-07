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

class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, hidden_dim, time_embed_dim, dropout=0.0):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize last layer to near zero for stable residuals
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)
    
    def forward(self, x, time_emb):
        # Add time conditioning
        h = x + self.time_mlp(time_emb)
        # Apply block with residual connection
        return x + self.block(h)
        
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
            time_embed_dim = 2 * config.model.nf
        else:
            self.time_embed = nn.Linear(1, config.model.nf)
            time_embed_dim = config.model.nf
        
        ####################################################
        # Adding residual blocks
        # Process time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, config.model.nf * 4),
            nn.SiLU(),
            nn.Linear(config.model.nf * 4, config.model.nf * 4)
        )
        ####################################################

        input_dim = 2
        hidden_dim = config.model.nf * 2
        
        # self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Better input processing with normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add normalization
            nn.SiLU()  # Better than ReLU
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, config.model.nf * 4, config.model.dropout)
            for _ in range(config.model.num_res_blocks)
        ])
                
        # self.hidden_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim + embed_dim, hidden_dim),
        #         nn.SiLU(),
        #         nn.Dropout(config.model.dropout)
        #     ) for _ in range(config.model.num_res_blocks)
        # ])
        # Modifying each layer so they get time conditioning + wider + normalized
        # self.hidden_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim + embed_dim, hidden_dim * 2),  # Expand
        #         nn.LayerNorm(hidden_dim * 2),
        #         nn.SiLU(),
        #         nn.Dropout(config.model.dropout),
        #         nn.Linear(hidden_dim * 2, hidden_dim),  # Contract back
        #         nn.LayerNorm(hidden_dim),
        #         nn.SiLU()
        #     ) for _ in range(config.model.num_res_blocks)
        # ])
        
        # self.output_layer = nn.Linear(hidden_dim, 2)
        # Output projection
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Initialize output layer to near-zero for stability
        nn.init.zeros_(self.output_layer[-1].weight)
        nn.init.zeros_(self.output_layer[-1].bias)
        
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Reshape from (batch, 2, 1, 1) to (batch, 2)
        if len(x.shape) == 4:
            x_flat = x.view(batch_size, 2)
        else:
            x_flat = x
            
        # Time embedding - ensure t is the right shape
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)
        elif len(t.shape) == 3:  # Sometimes t comes as (batch_size, 1, 1)
            t = t.view(batch_size, -1)  # Flatten to (batch_size, 1)
            
        time_emb = self.time_embed(t)  # Should be (batch_size, embed_dim)
        
        # Ensure time_emb is 2D
        if len(time_emb.shape) > 2:
            time_emb = time_emb.view(batch_size, -1)
        
        # Process time through MLP
        time_emb = self.time_mlp(time_emb)
        
        # Forward pass
        # h = torch.relu(self.input_layer(x_flat))  # (batch_size, hidden_dim)
        h = self.input_layer(x_flat) # without residual connections
        
        # for layer in self.hidden_layers:
        #     # Debug prints to understand tensor shapes
        #     # print(f"h shape: {h.shape}, time_emb shape: {time_emb.shape}")
        #     h_with_time = torch.cat([h, time_emb], dim=1)  # (batch_size, hidden_dim + embed_dim)
        #     # h = h + layer(h_with_time)  # Residual connection
        #     h = layer(h_with_time) # no residual

        # Apply residual blocks
        for block in self.residual_blocks:
            h = block(h, time_emb)
            
        score = self.output_layer(h)  # (batch_size, 2)
        
        # Reshape back to (batch, 2, 1, 1) if needed
        if len(x.shape) == 4:
            score = score.view(batch_size, 2, 1, 1)
            
        return score