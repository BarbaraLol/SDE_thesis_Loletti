"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov  

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G


class LinearSDE(SDE):
  """Linear SDE: dx_t = -Ax_t dt + ε dw_t
  
  This implements a linear SDE with time-independent drift matrix A and 
  diffusion coefficient ε. The non-equilibrium condition [A, A^T] ≠ 0 
  ensures the system doesn't satisfy detailed balance.
  
  Args:
    A_matrix: 2x2 matrix defining the linear dynamics
    epsilon: noise strength parameter (time-independent)
    N: number of discretization steps
  """

  def __init__(self, A_matrix, epsilon, N):
    super().__init__(N)
        
    # Convert A_matrix to proper tensor and numpy array
    if isinstance(A_matrix, list):
        A_matrix = np.array(A_matrix, dtype=np.float32)
    
    self.A = torch.tensor(A_matrix, dtype=torch.float32)
    self.A_np = A_matrix.astype(np.float64)  # Use float64 for numerical stability
    self.epsilon = epsilon
    
    # Verify non-equilibrium condition: [A, A^T] ≠ 0
    A_tensor = torch.tensor(A_matrix, dtype=torch.float32)
    commutator = torch.matmul(A_tensor, A_tensor.T) - torch.matmul(A_tensor.T, A_tensor)
    is_non_equilibrium = not torch.allclose(commutator, torch.zeros_like(commutator), atol=1e-6)
    
    if not is_non_equilibrium:
        print("Warning: Matrix A satisfies [A, A^T] = 0 (equilibrium system)")
    else:
        print(f"Non-equilibrium verified: [A, A^T] has norm {torch.norm(commutator):.4f}")
    
    # Pre-compute steady-state covariance Σ_∞
    # Solves: A Σ_∞ + Σ_∞ A^T = ε² I
    try:
        self.Sigma_inf = solve_continuous_lyapunov(self.A_np, self.epsilon**2 * np.eye(2))
        
        # Verify solution is positive definite
        eigvals = np.linalg.eigvals(self.Sigma_inf)
        if not np.all(eigvals > 0):
            raise ValueError("Steady-state covariance has non-positive eigenvalues")
        
        # Verify Lyapunov equation
        residual = self.A_np @ self.Sigma_inf + self.Sigma_inf @ self.A_np.T - self.epsilon**2 * np.eye(2)
        residual_norm = np.linalg.norm(residual)
        
        if residual_norm > 1e-6:
            print(f"Warning: Lyapunov equation residual = {residual_norm:.2e}")
        
        print(f"Steady-state covariance Σ_∞:\n{self.Sigma_inf}")
        print(f"Eigenvalues: {eigvals}")
        
    except Exception as e:
        print(f"Error solving Lyapunov equation: {e}")
        print("Using fallback: Σ_∞ = ε² I")
        self.Sigma_inf = (self.epsilon**2) * np.eye(2)

  @property
  def T(self):
    """End time of the SDE."""
    return 1.0

  def sde(self, x, t):
    """SDE dynamics: dx_t = -Ax_t dt + ε dw_t
    
    Since A and ε are time-independent, the drift and diffusion
    don't explicitly depend on time t.
    
    Args:
      x: state tensor (batch_size, 2, H, W)
      t: time tensor (batch_size,) - not used in time-independent case

    Returns:
      drift: -Ax in same format as x
      diffusion: ε for each batch element
    """
    batch_size = x.shape[0]
    A = self.A.to(x.device)
    
    # Reshape for matrix multiplication: (batch, 2, H, W) -> (batch, 2)
    x_flat = x.reshape(batch_size, 2)
    
    # Drift: -Ax
    drift_flat = -torch.matmul(x_flat, A.T)  # (batch, 2)
    drift = drift_flat.view(x.shape)
    
    # Diffusion: ε (constant)
    diffusion = torch.full((batch_size,), self.epsilon, device=x.device)
    
    return drift, diffusion

  def marginal_prob(self, x, t):
    """
    Exact marginal probability p_t(x|x_0) for dx_t = -Ax_t dt + ε dw_t
    
    For time-independent A and ε, the solution is:
      x_t = exp(-At) x_0 + ∫₀ᵗ exp(-A(t-s)) ε dw_s
    
    Therefore:
      Mean: μ(t) = exp(-At) x_0
      Covariance: Σ(t) = Σ_∞ - exp(-At) Σ_∞ exp(-A^T t)
    
    where Σ_∞ satisfies A Σ_∞ + Σ_∞ A^T = ε² I.
    
    This formula is EXACT for time-independent linear SDEs.
    """
    batch_size = x.shape[0]
    A_np = self.A_np
    epsilon_sq = self.epsilon ** 2
    
    x_flat = x.reshape(batch_size, 2).detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    
    means = []
    stds = []
    
    for i in range(batch_size):
      t_val = float(t_np[i])
      x0 = x_flat[i]
      
      if t_val < 1e-8:
        mean_i = x0
        std_i = np.full(2, 1e-8)
      else:
        # Mean: exp(-At) x_0
        exp_neg_At = expm(-A_np * t_val)
        mean_i = exp_neg_At @ x0
        
        # Covariance: ANALYTICAL FORMULA (exact and fast)
        # Σ(t) = Σ_∞ - exp(-At) Σ_∞ exp(-A^T t)
        exp_neg_ATt = expm(-A_np.T * t_val)
        
        Sigma_t = self.Sigma_inf - exp_neg_At @ self.Sigma_inf @ exp_neg_ATt
        
        # Extract standard deviations
        variances = np.diag(Sigma_t)
        # Ensure numerical stability
        variances = np.maximum(variances, 1e-8)
        std_i = np.sqrt(variances)
      
      means.append(torch.tensor(mean_i, device=x.device, dtype=torch.float32))
      stds.append(torch.tensor(std_i, device=x.device, dtype=torch.float32))
    
    mean_flat = torch.stack(means, dim=0)
    std_flat = torch.stack(stds, dim=0)
    
    mean = mean_flat.view(x.shape)
    std = std_flat.view(x.shape)
    
    return mean, std

  def prior_sampling(self, shape):
    """Generate samples from the prior distribution p_T(x).
    
    At t=T, the distribution approaches the steady state N(0, Σ_∞).
    For non-equilibrium systems, Σ_∞ is NOT isotropic.
    
    Args:
      shape: desired shape (batch_size, 2, H, W)
      
    Returns:
      samples from N(0, Σ_∞)
    """
    batch_size = shape[0]
    
    # Sample from N(0, Σ_∞) using Cholesky decomposition
    # Σ_∞ = L L^T, so if z ~ N(0,I), then Lz ~ N(0, Σ_∞)
    try:
        L = np.linalg.cholesky(self.Sigma_inf)
        z = torch.randn(batch_size, 2)
        samples_2d = torch.matmul(z, torch.tensor(L.T, dtype=torch.float32))
    except np.linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed, using fallback")
        # Fallback to isotropic sampling
        samples_2d = torch.randn(batch_size, 2) * self.epsilon
    
    # Reshape to match expected format
    if len(shape) == 4:  # (batch, 2, H, W)
        samples = samples_2d.view(batch_size, 2, 1, 1)
    else:
        samples = samples_2d
    
    return samples

  def prior_logp(self, z):
    """Compute log-density of the prior distribution.
    
    For z ~ N(0, Σ_∞):
      log p(z) = -0.5 * z^T Σ_∞^{-1} z - 0.5 * log|2π Σ_∞|

    Args:
      z: latent code
      
    Returns:
      log probability density for each sample in batch
    """
    z_flat = z.view(z.shape[0], 2)
    
    # Compute Σ_∞^{-1}
    try:
        Sigma_inv = torch.tensor(
            np.linalg.inv(self.Sigma_inf), 
            device=z.device, 
            dtype=torch.float32
        )
    except np.linalg.LinAlgError:
        print("Warning: Matrix inversion failed, using identity")
        Sigma_inv = torch.eye(2, device=z.device) / (self.epsilon**2)
    
    # Quadratic form: z^T Σ^{-1} z
    quadratic = torch.sum(z_flat * torch.matmul(z_flat, Sigma_inv), dim=1)
    
    # Log determinant term: log|2π Σ_∞|
    sign, logdet = np.linalg.slogdet(2 * np.pi * self.Sigma_inf)
    if sign <= 0:
        print("Warning: Determinant not positive, using fallback")
        logdet = 2 * np.log(2 * np.pi * self.epsilon**2)
    
    logp = -0.5 * quadratic - 0.5 * logdet
    
    return logp

  def discretize(self, x, t):
    """Euler-Maruyama discretization.
    
    Converts SDE to discrete-time update:
      x_{i+1} = x_i + f_i dt + g_i √dt z_i
    
    where z_i ~ N(0,1).
    
    Args:
      x: current state
      t: current time
      
    Returns:
      f: drift term (scaled by dt)
      G: diffusion term (scaled by √dt)
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=x.device))
    
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE for sampling.
    
    The reverse SDE is:
      dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw  (SDE)
      dx = [f(x,t) - 0.5 g(t)² ∇_x log p_t(x)] dt        (ODE)
    
    For our forward SDE dx_t = -Ax_t dt + ε dw_t:
      f(x,t) = -Ax
      g(t) = ε
    
    Args:
      score_fn: trained score model that estimates ∇_x log p_t(x)
      probability_flow: if True, return deterministic ODE instead of SDE
      
    Returns:
      RSDE: reverse SDE/ODE class instance
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize
    parent_A = self.A.clone()
    parent_epsilon = self.epsilon
    parent_A_np = self.A_np.copy()

    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow
        self.A = parent_A
        self.epsilon = parent_epsilon
        self.A_np = parent_A_np
        # Store Sigma_inf if parent has it
        if hasattr(parent_A_np, '__self__'):
          self.Sigma_inf = parent_A_np.__self__.Sigma_inf

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """
        IMPORTANT: In reverse sampling, t represents the current time
        going from T → 0, but we need the forward drift at this time.
        
        The forward SDE at time t is: dx = -Ax dt + ε dw
        The reverse SDE at time t is: dx = [-Ax - g²∇log p_t(x)] dt + g dw
        
        Note: Some implementations use "reverse time" τ = T - t, but
        the score function should receive the actual forward time.
        """
        # Get forward drift at current time t
        drift, diffusion = sde_fn(x, t)
        
        # Get score at current time t
        score = score_fn(x, t)
        
        # Reverse drift formula
        score_coefficient = 0.5 if self.probability_flow else 1.0
        diffusion_squared = diffusion[:, None, None, None] ** 2
        
        # KEY: reverse_drift = drift - g² * score
        # Since score = -Σ^(-1)(x-μ), this becomes:
        # reverse_drift = -Ax - g²(-Σ^(-1)(x-μ)) = -Ax + g²Σ^(-1)(x-μ)
        reverse_drift = drift - diffusion_squared * score_coefficient * score
        
        reverse_diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
        
        return reverse_drift, reverse_diffusion

      def discretize(self, x, t):
        f, G = discretize_fn(x, t)
        score = score_fn(x, t)
        
        score_coeff = 0.5 if self.probability_flow else 1.0
        rev_f = f - G[:, None, None, None]**2 * score * score_coeff
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        
        return rev_f, rev_G

    return RSDE()