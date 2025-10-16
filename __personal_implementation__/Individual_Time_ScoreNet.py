"""
SDE integration where each point has its own random terminal time
Forward: Point i evolves from t=0 to t=T_i
Backward: Point i evolves from t=T_i back to t=0
"""

import numpy as np
from typing import List, Tuple, Callable
from tqdm import tqdm
from SDE_integrator import SDESolver, SDEconfig


class IndividualTimeSDESolver:
    """
    SDE solver where each point has its own random terminal time
    """
    def __init__(self, config: SDEconfig, drift_fn: Callable, diffusion_matrix=None):
        self.base_solver = SDESolver(config, drift_fn, diffusion_matrix)
        self.config = config
    
    def forward_individual_times(self, 
                                 x0: np.ndarray, 
                                 min_time: float = None,
                                 max_time: float = None,
                                 save_every: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Forward SDE where each point evolves to its own random terminal time
        
        Args:
            x0: Initial positions (n_points, dim)
            min_time: Minimum terminal time (default: T/2)
            max_time: Maximum terminal time (default: T)
            save_every: Save trajectory every N steps
            
        Returns:
            trajectories: List of n_points trajectories, each with shape (n_steps_i, dim)
            times: List of n_points time arrays, each with shape (n_steps_i,)
            terminal_times: Array of terminal times for each point (n_points,)
        """
        n_points = x0.shape[0]
        
        # Sample random terminal times for each point
        if min_time is None:
            min_time = self.config.T / 2
        if max_time is None:
            max_time = self.config.T
        
        terminal_times = np.random.uniform(min_time, max_time, n_points)
        
        print(f"\n{'='*70}")
        print(f"Forward SDE with Individual Terminal Times")
        print(f"{'='*70}")
        print(f"  • Points: {n_points}")
        print(f"  • Terminal times range: [{terminal_times.min():.2f}, {terminal_times.max():.2f}]")
        print(f"  • Mean terminal time: {terminal_times.mean():.2f}")
        print(f"  • dt: {self.config.dt}")
        
        trajectories = []
        time_arrays = []
        
        # Evolve each point individually with progress bar
        for i in tqdm(range(n_points), desc="Forward integration"):
            x_start = x0[i:i+1]  # (1, dim)
            T_i = terminal_times[i]
            
            # Calculate number of steps for this point
            n_steps_i = int(T_i / self.config.dt)
            
            trajectory_i = [x_start.copy()]
            times_i = [0.0]
            
            x_current = x_start.copy()
            
            # Integrate from 0 to T_i
            for step in range(n_steps_i):
                t = step * self.config.dt
                x_current = self.base_solver.forward_step(x_current, t)
                
                if (step + 1) % save_every == 0:
                    trajectory_i.append(x_current.copy())
                    times_i.append(t + self.config.dt)
            
            # Convert to arrays
            trajectory_i = np.array(trajectory_i).squeeze()  # (n_steps_i, dim)
            times_i = np.array(times_i)
            
            trajectories.append(trajectory_i)
            time_arrays.append(times_i)
        
        print(f"{'='*70}\n")
        
        return trajectories, time_arrays, terminal_times
    
    def backward_individual_times(self,
                                  final_positions: np.ndarray,
                                  terminal_times: np.ndarray,
                                  score_fn: Callable,
                                  save_every: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward SDE where each point evolves from its own terminal time back to 0
        
        Args:
            final_positions: Final positions at terminal times (n_points, dim)
            terminal_times: Terminal time for each point (n_points,)
            score_fn: Learned score function
            save_every: Save trajectory every N steps
            
        Returns:
            trajectories: List of n_points trajectories, each with shape (n_steps_i, dim)
            times: List of n_points time arrays, each with shape (n_steps_i,)
        """
        n_points = final_positions.shape[0]
        
        print(f"\n{'='*70}")
        print(f"Backward SDE with Individual Terminal Times")
        print(f"{'='*70}")
        print(f"  • Points: {n_points}")
        print(f"  • Terminal times range: [{terminal_times.min():.2f}, {terminal_times.max():.2f}]")
        
        trajectories = []
        time_arrays = []
        
        # Evolve each point backward individually with progress bar
        for i in tqdm(range(n_points), desc="Backward integration"):
            x_T_i = final_positions[i:i+1]  # (1, dim)
            T_i = terminal_times[i]
            
            # Calculate number of steps for this point
            n_steps_i = int(T_i / self.config.dt)
            
            trajectory_i = [x_T_i.copy()]
            times_i = [T_i]
            
            x_current = x_T_i.copy()
            
            # Integrate backward from T_i to 0
            for step in range(1, n_steps_i + 1):
                t_curr = T_i - (step - 1) * self.config.dt
                x_current = self.base_solver.backward_step(x_current, t_curr, score_fn)
                
                if step % save_every == 0:
                    trajectory_i.append(x_current.copy())
                    times_i.append(T_i - step * self.config.dt)
            
            # Convert to arrays
            trajectory_i = np.array(trajectory_i).squeeze()  # (n_steps_i, dim)
            times_i = np.array(times_i)
            
            trajectories.append(trajectory_i)
            time_arrays.append(times_i)
        
        print(f"{'='*70}\n")
        
        return trajectories, time_arrays
    
    def create_training_dataset(self,
                               trajectories: List[np.ndarray],
                               time_arrays: List[np.ndarray],
                               subsample: int = 1) -> List[Tuple[np.ndarray, float]]:
        """
        Create training dataset from individual trajectories
        
        Args:
            trajectories: List of trajectories for each point
            time_arrays: List of time arrays for each point
            subsample: Use every Nth point from each trajectory
            
        Returns:
            dataset: List of (position, time) tuples
        """
        dataset = []
        
        print(f"\nCreating training dataset...")
        for traj, times in tqdm(zip(trajectories, time_arrays), total=len(trajectories), desc="Processing trajectories"):
            # Subsample trajectory
            for j in range(0, len(traj), subsample):
                pos = traj[j]
                t = times[j]
                dataset.append((pos, t))
        
        print(f"✓ Training dataset created: {len(dataset)} samples\n")
        return dataset

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
              use_scheduler: bool = False,
              scheduler_type: str = 'cosine',
              warmup_epochs: int = 0,
              grad_clip: float = None,
              save_best: bool = True,
              checkpoint_dir: str = 'checkpoints',
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
            use_scheduler: Whether to use learning rate scheduling
            scheduler_type: 'cosine' or 'step'
            warmup_epochs: Number of warmup epochs before scheduling
            grad_clip: Gradient clipping value (None = no clipping)
            save_best: Save best model during training
            checkpoint_dir: Directory for checkpoints
            verbose: Print progress
            
        Returns:
            loss_history: List of average losses per epoch
        """
        # Create checkpoint directory
        if save_best:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        
        optimizer = optim.AdamW(self.score_net.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = None
        if use_scheduler:
            if scheduler_type == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)
            elif scheduler_type == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//4, gamma=0.5)
            elif scheduler_type == 'exponential':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        loss_history = []
        best_loss = float('inf')
        
        # Create dataset: (x, t) pairs from all trajectory snapshots
        dataset = []
        for positions, time in zip(self.trajectory, self.times):
            for pos in positions:
                dataset.append((pos, time))  # NO time normalization
        
        n_data = len(dataset)
        
        print(f"\nTraining Configuration:")
        print(f"  • Dataset: {n_data} points")
        print(f"  • Epochs: {n_epochs}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Learning rate: {lr}")
        print(f"  • Denoising σ: {sigma_dn}")
        print(f"  • Weight decay: {weight_decay}")
        print(f"  • Scheduler: {scheduler_type if use_scheduler else 'None'}")
        print(f"  • Grad clip: {grad_clip if grad_clip else 'None'}")
        print(f"  • Save best: {save_best}")
        print(f"  • Device: {self.device}\n")
        
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
                
                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), grad_clip)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            
            # Save best model
            if save_best and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.score_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'dim': self.dim,
                }, best_model_path)
                if verbose and epoch > 0:
                    print(f"\n💾 Best model saved! Loss: {best_loss:.6f}")
            
            # Update learning rate
            if scheduler is not None:
                if epoch >= warmup_epochs:
                    scheduler.step()
            
            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{avg_loss:.6f}', 
                    'best': f'{best_loss:.6f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Print periodic updates
            if (epoch + 1) % 100 == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}, Best: {best_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save final model
        if save_best:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.score_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history[-1],
                'dim': self.dim,
            }, final_model_path)
            print(f"\n Final model saved to: {final_model_path}")
            print(f" Best model saved to: {best_model_path}")
            print(f"   Best loss: {best_loss:.6f}")
        
        self.score_net.eval()
        print(f"\n✓ Training completed! Final loss: {loss_history[-1]:.6f}")
        return loss_history
    
    def load_best_model(self, checkpoint_dir: str = 'checkpoints'):
        """Load the best model from checkpoints"""
        import os
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.score_net.load_state_dict(checkpoint['model_state_dict'])
            self.score_net.eval()
            print(f"✓ Best model loaded from {best_model_path}")
            print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
            return checkpoint['loss']
        else:
            print(f"⚠ Best model not found at {best_model_path}")
            return None
    
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