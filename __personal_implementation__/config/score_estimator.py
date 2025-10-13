import numpy as np
from typing import List, Callable, Tuple
from scipy.spatial.distance import cdist

class GaussianKDEEstimator:
    '''
    Score function ∇log p_t(x) estimation using the KDE (kernel density estimation) with a gaussian kernel and the data coming from the forward trajectory
    '''
    def __init__(self, forward_trajectory: List[np.ndarray], forward_time: np.ndarray, bandwidth: float = None, bandwidth_method: str = 'scott', 
                    k_neighbors: int = None):
        '''
        Args
            forward_trajectory: List of arrays (n_points, dim) resulting from the forward SDE
            forward_time: Arrays of times coming corresponding to trajectory snapshots
            bandwidth: KDE bandwidth sigma. If None, computed automatically
            badwidth_method: 'scott', 'silverman', or 'fixed'
            k_neighbors: If provided, use only K nearest neighbors (K-NN KDE)
        '''
        self.trajectory = forward_trajectory
        self.time = forward_time
        self.dim = forward_trajectory[0].shape[1]
        self.k_neighbors = k_neighbors

        # Computing the bandwidth
        if bandwidth is None:
            self.bandwidth = self._compute_bandwidth(bandwidth_method)
        else:
            self.bandwidth = bandwidth

    def _compute_bandwidth(self, method: str) -> float:
        '''
        Function to compute the bandwidth using the standard rules

        Args:
            method: either 'scott' or 'silverman'

        Returns:
            bandwidth sigma
        '''
        # Using the data from all the time slices so to estimate
        all_data = np.vstack(self.trajectory)
        n = all_data.shape[0]
        d = all_data.shape[1]

        # Standard deviation of the data
        std = np.std(all_data, axis = 0).mean()

        if method == 'scott':
            # Using Scott method
            # Sigma = n^(-1/(d+4)) * std
            bandwidth = (n ** (-1.0 / (d + 4.0))) * std
        elif method == 'silverman':
            # Using Silverman method
            # Sigma (n * (d + 2) / 4)^(-1/(d+4)) * std
            bandwidth = ((n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))) *std
        else:       
            raise ValueError(f"Unknown bandwidth method: {method}")
        
        return bandwidth

    def _find_closest_time_index(self, t: float) -> int:
        '''Function to find the closest time in the forward diffusion/trajectory'''
        idx = np.argmin(np.abs(self.time - t))
        return idx

    def estimate_score(self, x: np.ndarray, t: float) -> np.ndarray:
        '''
        Score function ∇log p_t(x) estimation using the KDE.

        For a Gaussian KDE the score function is
            ∇log p(x) = Σ_i w_i * (x_i - x) / sigma^2
        with w_i are normalized kernel weights

        Args:
            x: query import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

@dataclass
class SDEconfig: 
    '''configurations'''
    dim: int # System dimentions
    T: float # Final time T
    dt: float # timestep
    epsilon: float = 0.1
    n_steps: int=None # In case it is not provided, it is computed by doing T/dt

    def __post_init__(self):
        if self.n_steps is None:
            self.n_steps = int(self.T / self.dt)

class SDESolver:
    '''
    General SDE/ODE solver for systems having this form
        dx_t = f(x_t, t) dt + epsilon * dW_t
    Where
        - f: drift function (deterministic dynamics)
        - epsilon: noise (set to 0 in case of ODE)
        dW_t: Brownian motion
    '''
    def __init__(self, config, drift_fn : Callable[[np.ndarray, float], np.ndarray], diffusion_matrix : Optional[np.ndarray] = None):
        '''
        Args:
            config: configuration file
            drift_fn: drift diffusion f(x, t), returning shape (n_points, dim)
            diffusion matrix G
        '''
        self.config = config
        self.drift_fn = drift_fn

        # Check if there have been inserted a diffusion matrix, otherwise use the default one
        if diffusion_matrix is None:
            self.diffusion_matrix = np.eye(config.dim)*config.epsilon # Base case of (epsilon)*I
        else:
            self.diffusion_matrix = diffusion_matrix

        # Square root of the diffusion with Cholesky for sampling
        # Computing the Cholesky factor L of $Sigma$ = G * G^T
        self.diffusion_sqrt = np.linalg.cholesky(self.diffusion_matrix @ self.diffusion_matrix.T)

    def forward_step(self, x : np.ndarray, t : float) -> np.ndarray:
        ''' 
        Single Euler-Murayama step (forward in time)

        Args:
            x: current position as a batch (n_points, dim)
            t: current time

        Returns:
            x_next: position of the points at time t+dt
        '''
        n_points = x.shape[0]
        dt = self.config.dt

        # drift term
        drift = self.drift_fn(x, t)

        # Computing the diffusion term: epsilon * dW 
        if self.config.epsilon > 0:
            dW = np.random.randn(n_points, self.config.dim)
            diffusion = (dW @ self.diffusion_sqrt.T) * np.sqrt(dt)
        else:
            diffusion = 0.0 # ODE case

        # Update using Euler-Murayama:
        x_next = x + drift * dt + diffusion

        return x_next

    def forward_trajectory(self, x0: np.ndarray, save_every: int = 1) -> Tuple[List[np.ndarray], np.ndarray]:
        '''
        Forward SDE integrator, from t = 0 to t = T

        Args:
            x0: Initial positions in the form of (n_points, dim)
            save_every: Save trajectory for the N steps

        Returns
            trajectory: A list of saved positions
            times: Array of saved times
        '''
        trajectory = [x0.copy()]
        times = [0.0]

        x = x0.copy()

        # Computing the trajctories 
        for step in range(self.config.n_steps):
            t = step * self.config.dt 
            x = self.forward_step(x, t) # Computing state at time t + dt

            if (step + 1) % save_every == 0:
                trajectory.append(x.copy())
                times.append(t + self.config.dt)

        return trajectory, np.array(times)   

    def backward_step(self, x: np.ndarray, t: float, score_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        '''
        Single reverse SDE step backward in time

        The reverse SDE is computed as dx_t = [f(x,t) - g^2∇log p_t(x)] dt + g*dW

        with g^2 = epsilon^2 is the diffusion coefficient

        Args:
            x: current (starting) position at time T and with shape (n_points, dim)
            t: current time (from T to 0)
            score_fn: model of the score function ∇log p_t(x)

        Returns:
            x_prev: Position at timestep t-dt
        '''
        n_points = x.shape[0]
        dt = self.config.dt

        # Get the score: ∇log p_t(x)
        score = score_fn(x, t)

        # Computing the reverse drift: f(x,t) - g²∇log p_t(x)
        # matrix diffusion or identity matrix cases distinction
        drift = self.drift_fn(x, t)
        # Anderson's coefficient: g = √(2)*ε
        g = self.config.epsilon * np.sqrt(2)
        reverse_drift = drift - (g**2) * score

        # Computing the reverse diffusion term
        if self.config.epsilon > 0:
            dW = np.random.randn(n_points, self.config.dim)
            diffusion = (dW @ self.diffusion_sqrt.T) * g * np.sqrt(dt)
        else:
            diffusion = 0.0 # ODE case

        # Backward Euler-Murayama (going backward in time)
        x_prev = x - reverse_drift * dt + diffusion

        return x_prev

    def backward_trajectory(self, x_T: np.ndarray, score_fn: Callable[[np.ndarray, float], np.ndarray], save_every: int = 1) -> Tuple[List[np.ndarray], np.ndarray]:
        '''
        Integrating the reverse SDE backward form t = T to t = 0

        Args:
            x_T: initial position at time T with shape (n_points, dim)
            score_fn: score function ∇log p_t(x)
            save_every: save the trajectory for all the N steps

        Return:
            trajectory: List of saved positions (from T to 0)
            times: Array of saved times (from T to 0)
        '''
        trajectory = [x_T.copy()]
        times = [self.config.T]

        x = x_T.copy()

         # Computing the reverse trajctories 
        for step in range(1, self.config.n_steps + 1):
            t_curr = self.config.T - (step - 1) * self.config.dt
            x = self.backward_step(x, t_curr, score_fn) # Computing state at time t - dt

            if step % save_every == 0:
                trajectory.append(x.copy())
                times.append(self.config.T - step * self.config.dt)

        return trajectory, np.array(times) 

        
(sampled) points (n_query, dim)
            t: time at which to estimate the score

        Returns:
            score: ∇log p_t(x) with shape (n_query, dim)
        '''
        # Get the data points at time t from the forward trajectory
        time_idx = self._find_closest_time_index(t)
        data_points = self.trajectory[time_idx] # (n_data, dim)

        n_query = x.shape[0]
        n_data = data_points.shape[0]

        # Compute pairwise squared distances: ||x - x_i||²
        # Shape: (n_query, n_data)
        sq_distances = cdist(x, data_points, metric='sqeuclidean')

        # If using K-NN, select only K nearest neighbors for each query
        if self.k_neighbors is not None and self.k_neighbors < n_data:
            knn_indices = np.argpartition(sq_distances, self.k_neighbors, axis=1)[:, :self.k_neighbors]
            
            # Create mask to zero out non-neighbors
            mask = np.zeros_like(sq_distances, dtype=bool)
            for i in range(n_query):
                mask[i, knn_indices[i]] = True
            
            # Set distances of non-neighbors to infinity (zero weight)
            sq_distances = np.where(mask, sq_distances, np.inf)

        # Compute unnormalized kernel weights: exp(-||x - x_i||^2 / (2sigma^2))
        sigma_sq = self.bandwidth ** 2
        log_weights = -sq_distances / (2 * sigma_sq)

        # Normalize weight using the log-sum-exp trick to achieve numerical stability
        max_log_weights = np.max(log_weights, axis = 1, keepdims = True)
        weights = np.exp(log_weights - max_log_weights) # (n_query, n_data)
        weights = weights / np.sum(weights, axis = 1, keepdims = True)

        # Compute the score through the Σ_i w_i * (x_i - x) / sigma^2
        # For each query point, compute weighted sum of directions to data points
        score = np.zeros_like(x)

        for i in range (n_query):
            # Computing the direction from x[i] to each data point, so x_j - x[i]
            directions = data_points - x[i:i+1]  # (n_data, dim)
            # Weighted sum
            score[i] = np.sum(weights[i:i+1].T * directions, axis=0) / sigma_sq
        
        return score
    
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        '''Making the estimatore collable just like the score_fn(x, t)'''
        return self.estimate_score(x, t)

class AdaptiveBandwidthKDEScoreEstimator(GaussianKDEEstimator):
    '''
    KDE Estimator with time dependent bandwidth.

    The idea is that as t → 0, the distribution becomes more concentrated and so we might want to decrease bandwidth.
    '''

    def __init__(self, forward_trajectory: List[np.ndarray], forward_time: np.ndarray, bandwidth_schedule: Callable[[float], float] = None, 
                base_bandwidth: float = None, k_neighbors: int = None):
        '''
        Args:
            forward_trajectory: List of arrays from forward SDE
            forward_time: Array of times
            bandwidth_schedule: Function sigma(t) that returns bandwidth at time t
            base_bandwidth: Base bandwidth (if None, computed automatically)
            k_neighbors: K nearest neighbors (optional)
        '''
        super().__init__(forward_trajectory, forward_time, bandwidth = base_bandwidth, bandwidth_method = 'scott', k_neighbors=k_neighbors)

        if bandwidth_schedule is None:
            # Default option: decreasing the bandwidth in a linear way as t -> 0
            T = forward_time[-1]
            self.bandwidth_schedule = lambda t: self.bandwidth * (0.3 + 0.7 * t / T)
        else:
            self.bandwidth_schedule = bandwidth_schedule

    def estimate_score(self, x: np.ndarray, t: float) -> np.ndarray:
        '''Score estimation but with time dependent bandwidth'''
        # Temporarily override bandwidth
        original_bandwidth = self.bandwidth
        self.bandwidth = self.bandwidth_schedule(t)
        
        # Call parent method
        score = super().estimate_score(x, t)
        
        # Restore original bandwidth
        self.bandwidth = original_bandwidth
        
        return score