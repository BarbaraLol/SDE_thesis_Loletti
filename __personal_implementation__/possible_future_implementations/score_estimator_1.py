"""
Simple Gaussian KDE Score Estimator
This implementation is done so to compute analytically the score function
(The model will later have to learn the ∇log p(x), instead of computing it analytically)
"""

import numpy as np
from typing import List
from scipy.spatial.distance import cdist

class SimpleKDEScoreEstimator:
    """
    Simple score function estimator using Gaussian KDE
    
    For a Gaussian KDE, the score is:
        ∇log p(x) ≈ Σ_i w_i * (x_i - x) / σ²
    where w_i measure how much each data point influences the score at query point x

    This is done because p̂(x) = (1/N) Σᵢ δ(x - xᵢ) (sum of Dirac deltas) is not smooth nor differentiable!
    """
    def __init__(self, 
                 forward_trajectory: List[np.ndarray],
                 forward_times: np.ndarray,
                 sigma: float = None):
        """
        Args:
            forward_trajectory: List of arrays (n_points, dim) from forward SDE
            forward_times: Array of times corresponding to trajectory
            sigma: KDE bandwidth sigma. If None, computed using Scott's rule
        """
        self.trajectory = forward_trajectory
        self.times = forward_times
        self.dim = forward_trajectory[0].shape[1]
        
        # Compute sigma if not provided
        if sigma is None:
            self.sigma = self._compute_sigma_scott()
        else:
            self.sigma = sigma
        
        print(f"KDE Score Estimator initialized:")
        print(f"  • Sigma sigma: {self.sigma:.4f}")
        print(f"  • Trajectory snapshots: {len(forward_trajectory)}")
        print(f"  • Dimension: {self.dim}")
    
    def _compute_sigma_scott(self) -> float:
        """
        Scott's rule comes from minimizing the Mean Integrated Squared Error (MISE) between the KDE and the true density, under the assumption that the 
        true density is Gaussian.

        Compute sigma using Scott's rule:
        sigma = n^(-1/(d+4)) * std(data)

            - n: number of data points
            - d: dimentions
        """
        all_data = np.vstack(self.trajectory)
        n = all_data.shape[0]
        d = all_data.shape[1]
        std = np.std(all_data, axis=0).mean()
        sigma = (n ** (-1.0 / (d + 4.0))) * std
        return sigma
    
    def _find_closest_time_index(self, t: float) -> int:
        """Find closest time snapshot in trajectory"""
        idx = np.argmin(np.abs(self.times - t))
        return idx
    
    def estimate_score(self, x: np.ndarray, t: float) -> np.ndarray:
        '''
        We want to estimate the score function:
            ∇ₓ log p_t(x)
        using a non-parametric Gaussian KDE with bandwidth sigma.

        The KDE is defined as:
            p̂_t(x) = (1 / N) * Σ_i exp( -‖x - x_i‖^2 / (2sigma^2) )

        Differentiating with respect to x gives:
            ∇ₓ log p̂_t(x)
                = [ Σ_i exp( -‖x - x_i‖^2 / (2sigma^2) ) * (x_i - x) / sigma^2 ] 
                / [ Σ_i exp( -‖x - x_i‖^2 / (2sigma^2) ) ]

        which can be rewritten as:
            ∇ₓ log p̂_t(x) = Σ_i w_i(x) * (x_i - x) / sigma^2

        where:
            w_i(x) = K_sigma(x, x_i) / Σ_j K_sigma(x, x_j)
        are the normalized (softmax) Gaussian kernel weights.

        In other words, the score points from the query point x 
        toward the kernel-weighted mean of nearby data points, 
        and its magnitude scales with 1/sigma^2. 
        This is exactly the formula implemented in the code.
        
        Args:
            x: query points (n_query, dim)
            t: time at which to estimate score
            
        Returns:
            score: ∇log p_t(x) with shape (n_query, dim)
        '''
        # ADD THE FACT THAT i HAVE TO SAMPLE FROM THE FORWARD TRAJECTORY
        # Get data points at time t
        time_idx = self._find_closest_time_index(t)
        data_points = self.trajectory[time_idx]  # (n_data, dim)
        
        # query-by-data algorithm, but with O(nquery ndata)
        n_query = x.shape[0]
        n_data = data_points.shape[0]
        
        # Computing p^~(t) via the kernel weights
        # Compute pairwise squared distances: ||x - x_i||^2
        sq_distances = cdist(x, data_points, metric='sqeuclidean')
        
        # Compute kernel weights: exp(-||x - x_i||^2 / (2sigma^2))
        log_weights = -sq_distances / (2 * (self.sigma ** 2))
        
        # Normalize weights using log-sum-exp trick for numerical stability
        max_log_weights = np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights - max_log_weights)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Compute score: 
        '''
        Conceptually:

            ∇ₓ log p̂_t(x^(q)) 
                = (1 / sigma^2) * [ Σ_i w_i(x^(q)) * x_i  -  x^(q) * Σ_i w_i(x^(q)) ]
                = [ (Σ_i w_i(x^(q)) * x_i)  -  x^(q) ] / sigma^2

        because the weights are normalized, i.e.:
            Σ_i w_i(x^(q)) = 1

        Thus, the score can be interpreted as the vector pointing
        from the query point x^(q) toward the weighted mean of 
        the nearby data points, scaled by 1 / sigma^2.
        '''
        score = np.zeros_like(x)
        for i in range(n_query):
            # Directions from x[i] to each data point
            directions = data_points - x[i:i+1]  # (n_data, dim)
            # Weighted sum
            score[i] = np.sum(weights[i:i+1].T * directions, axis=0) / (self.sigma ** 2)
        
        return score
    
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Make the estimator callable like score_fn(x, t)"""
        return self.estimate_score(x, t)


def create_kde_score_estimator(forward_trajectory: List[np.ndarray],
                               forward_times: np.ndarray,
                               sigma: float = None):
    """
    Factory function to create a KDE score estimator
    
    Args:
        forward_trajectory: List of arrays from forward SDE
        forward_times: Times corresponding to snapshots
        sigma: Optional sigma (if None, auto-computed)
        
    Returns:
        score_fn: Callable score function (x, t) -> score
    """
    estimator = SimpleKDEScoreEstimator(
        forward_trajectory=forward_trajectory,
        forward_times=forward_times,
        sigma=sigma
    )
    
    return estimator