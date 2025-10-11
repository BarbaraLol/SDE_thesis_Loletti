import numpy as np
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
        g_squared = self.config.epsilon ** 2
        reverse_drift = drift - g_squared * score

        # Computing the reverse diffusion term
        if self.config.epsilon > 0:
            dW = np.random.randn(n_points, self.config.dim)
            diffusion = (dW @ self.diffusion_sqrt.T) * np.sqrt(dt)
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
        for step in range(self.config.n_steps):
            t = step * self.config.dt 
            x = self.backward_step(x, t, score_fn) # Computing state at time t - dt

            if (step + 1) % save_every == 0:
                trajectory.append(x.copy())
                times.append(t - self.config.dt)

        return trajectory, np.array(times) 

        
