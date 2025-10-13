import numpy as np
from typing import List, Callable, Tuple
from scipy.spatial.distance import cdist

class AdaptiveGaussianScoreEstimator:
    '''
    Score estimator using the gaussian kernel (i.e.: adaptive Gaussian Mixture Mode)

    Steps:
        1. For every currently sampled point, it generates N points using local gaussians
        2. Combining everything in a single distribution p_t
        3. Score estimation from p_t
    '''

    def __init__(self, n_samples_per_point: int = 100, local_std: float = 0.08, bandwidth: float = None, bandwidth_method: str = 'scott'):
        '''
        Args:
            - n_samples_per_point: number of points generated (with a gaussian) for every starting point of the backward sample
            - local_std: Local gaussians' stndard deviation
            - bandwidth: KDE bandwidth (in None, it is automatically computed with a specific function)
            - bandwidth_method: to chose between 'scott' and 'silverman'
        '''
        self.n_samples_per_point = n_samples_per_point
        self.local_std = local_std
        self.base_bandwidth = bandwidth
        self.bandwidth_method = bandwidth_method
        self.current_distribution = None # it referres to the current p_t distribution

        # To storage the distributions at different times
        self.distributions = {} # {t: samples}
        self.bandwidths = {}     # {t: bandwidth}

    def _generate_local_samples(self, points, radius = 0.1) -> np.ndarray:
        '''
        Uniform sample generation in a sphere of radius r centered on each single point
        '''
        n_points, dim = points.shape

        # Generating N n_samples_per_point (for every point) 
        samples_list = []
        for point in points:
            # Generating casual directions 
            directions = np.random.randn(self.n_samples_per_point, dim)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)

            # Casual radius (uniform in the volume)
            radii = np.random.rand(self.n_samples_per_point) ** (1/dim) * radius

            # Uniformly sampling in the circus/sphere
            local_samples = point + directions * radii[:, np.newaxis]
            samples_list.append(local_samples)

        # Combining everything in a single array
        all_samples = np.vstack(samples_list)

        return all_samples
    
    def _compute_bandwidth(self, data: np.ndarray) -> float:
        '''Computing the bandwidth based on the current data'''
        n = data.shape[0]
        d = data.shape[1]
        std = np.std(data, axis = 0).mean()

        if self.bandwidth_method == 'scott':
            # Using Scott method
            # Sigma = n^(-1/(d+4)) * std
            return (n ** (-1.0 / (d + 4.0))) * std
        elif self.bandwidth_method == 'silverman':
            # Using Silverman method
            # Sigma (n * (d + 2) / 4)^(-1/(d+4)) * std
            return ((n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))) * std
        else:       
            # raise ValueError(f"Unknown bandwidth method: {method}")
            return std * 0.5 # Default value

    # Da rivedere
    # def update_distribution(self, current_points: np.ndarray, t: float):
    #     '''
    #     Updating the current p_t distribution by generating gaussian samples near the current points

    #     Args:
    #         current_points: current points of the backward diffusion process (n_points, dim)
    #     '''
    #     # Generating Gaussian Mixture Models from current points
    #     samples = self._generate_local_samples(current_points)

    #     # Saving the distribution at time t
    #     self.distributions[t] = samples

    #     # Bandwidth update based on the new local distribution
    #     if self.base_bandwidth is None:
    #         self.bandwidths[t] = self._compute_bandwidth(samples)
    #     else:
    #         self.bandwidths[t] = self.base_bandwidth

    def estimate_score(self, x: np.ndarray, t: float) -> np.ndarray:
        '''
        Score function ∇log p_t(x) estimation using the KDE.

        For a Gaussian KDE the score function is
            ∇log p(x) = Σ_i w_i * (x_i - x) / h^2
        with 
            w_i(x) = K_h(x - x_i) / Σ_j K_h(x - x_j)
            K_h(u) = exp(-||u||^2/(2h^2))

        Args:
            x: query (sampled) points (n_query, dim)
            t: time at which to estimate the score

        Returns:
            score: ∇log p_t(x) with shape (n_query, dim)
        '''
        # Looking for the nearest timestep for which we know the distribution
        if t not in self.distributions:
            available_times = list(self.distributions.keys())
            if len(available_times) == 0:
                raise ValueError(f"No distribution available! Call update_distribution first.")

            # Looking for the nearest time availabele
            closest_t = min(available_times, key = lambda t_val: abs(t_val - t))
            data_points = self.distributions[closest_t]
            bandwidth = self.bandwidths[closest_t]
        else:
            data_points = self.distributions[t]
            bandwidth = self.bandwidths[t]

        n_query = x.shape[0]
        n_data = data_points.shape[0]

        # Compute pairwise squared distances: ||x - x_i||^2
        # Shape: (n_query, n_data)
        sq_distances = cdist(x, data_points, metric='sqeuclidean')

        # Compute kernel weights: K_h(x - x_i) = exp(-||x - x_i||^2 / (h^2))
        h_sq = bandwidth ** 2
        log_weights = -sq_distances / (2 * h_sq)

        # Normalize weights using the log-sum-exp trick to achieve numerical stability
        max_log_weights = np.max(log_weights, axis = 1, keepdims = True)
        weights = np.exp(log_weights - max_log_weights) # (n_query, n_data)
        weights = weights / np.sum(weights, axis = 1, keepdims = True)

        # Compute the score through the ∇log p_t(x) = Σ_i w_i * (x_i - x) / h^2
        # For each query point, compute weighted sum of directions to data points
        score = np.zeros_like(x)

        for i in range (n_query):
            # Computing the direction from x[i] to each data point, so x_j - x[i]
            directions = data_points - x[i:i+1]  # x_i - x
            # Weighted sum
            score[i] = np.sum(weights[i:i+1].T * directions, axis=0) / h_sq
        
        return score

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        '''
        Making the score_fn(x, t) a callable object.

        IMPORTANT: The score implicitly depends from the time t!
        '''
        return self.estimate_score(x, t)

    def get_distribution_at_time(self, t: float) -> np.ndarray:
        ''' It returns the samples of the distribution at time t'''
        if t in self.distributions:
            return self.distributions[t]
        
        # Trova tempo più vicino
        available_times = list(self.distributions.keys())
        if len(available_times) == 0:
            return None
        closest_t = min(available_times, key=lambda t_val: abs(t_val - t))
        return self.distributions[closest_t]
    

def backward_trajectory_adaptive(solver, x_T: np.ndarray,
                                 score_estimator: AdaptiveGaussianScoreEstimator,
                                 save_every: int = 1,
                                 verbose: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """Backward trajectory with adaptive score"""
    trajectory = [x_T.copy()]
    times = [solver.config.T]
    
    x = x_T.copy()
    
    if verbose:
        print(f"\n[Backward SDE with Adaptive Score]")
        print(f"  Starting from t={solver.config.T:.2f}")
        print(f"  Total steps: {solver.config.n_steps}")
    
    for step in range(solver.config.n_steps):
        t_current = solver.config.T - step * solver.config.dt
        
        # Update distribution p_t
        score_estimator.update_distribution(x, t_current)
        
        # Backward step
        x = solver.backward_step(x, t_current, score_estimator)
        
        if (step + 1) % save_every == 0:
            trajectory.append(x.copy())
            times.append(solver.config.T - (step + 1) * solver.config.dt)
            
            if verbose and (step + 1) % max(1, solver.config.n_steps // 10) == 0:
                progress = (step + 1) / solver.config.n_steps * 100
                print(f"  Progress: {progress:.0f}% | t={times[-1]:.2f}")
    
    if verbose:
        print(f"  ✓ Completed!")
    
    return trajectory, np.array(times)


def run_adaptive_backward(config, drift_fn, x0, args):
    """Run forward + adaptive backward simulation"""
    from SDE_integrator import SDEconfig, SDESolver
    import visualization
    
    sde_config = SDEconfig(
        dim=config.data.dim,
        T=config.data.T,
        dt=config.data.dt,
        epsilon=config.data.epsilon
    )
    
    solver = SDESolver(sde_config, drift_fn)
    
    print(f"\n{'='*70}")
    print(f"  ADAPTIVE BACKWARD SIMULATION")
    print(f"{'='*70}")
    
    # FORWARD
    print(f"\n{'→'*35}")
    print(f"  FORWARD SDE")
    print(f"{'→'*35}")
    
    save_every = config.visualization.save_every
    forward_traj, forward_times = solver.forward_trajectory(x0, save_every=save_every)
    
    print(f"✓ Forward completed: {len(forward_traj)} snapshots")
    
    if not args.no_plot:
        visualization.plot_forward_trajectories(forward_traj, forward_times,
                                               title=f"Forward: {config.name}")
    
    # BACKWARD
    print(f"\n{'←'*35}")
    print(f"  BACKWARD SDE")
    print(f"{'←'*35}")
    
    score_estimator = AdaptiveGaussianScoreEstimator(
        n_samples_per_point=args.n_samples_per_point,
        local_std=args.local_std,
        bandwidth=getattr(args, 'bandwidth', None),
        bandwidth_method='scott'
    )
    
    x_T = forward_traj[-1]
    
    backward_traj, backward_times = backward_trajectory_adaptive(
        solver, x_T, score_estimator,
        save_every=save_every,
        verbose=True
    )
    
    print(f"✓ Backward completed: {len(backward_traj)} snapshots")
    
    # Statistics
    x_reconstructed = backward_traj[-1]
    errors = np.linalg.norm(x0 - x_reconstructed, axis=1)
    
    print(f"\n{'='*70}")
    print(f"  RECONSTRUCTION STATISTICS")
    print(f"{'='*70}")
    print(f"  Mean:   {np.mean(errors):.4f}")
    print(f"  Median: {np.median(errors):.4f}")
    print(f"  Max:    {np.max(errors):.4f}")
    print(f"  Min:    {np.min(errors):.4f}")
    print(f"{'='*70}\n")
    
    if not args.no_plot:
        visualization.plot_forward_backward_comparison(
            forward_traj, forward_times,
            backward_traj, backward_times,
            x0,
            title=f"Adaptive Backward: {config.name}"
        )
    
    return forward_traj, forward_times, backward_traj, backward_times