''' Different drift functions'''

import numpy as np

# Linear drift with matrix A
def linear_drift(x: np.ndarray, t: float, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    '''
    Linear drift function: dx/dt = A·x
    
    Matrix A = [[-a,  b ],
                [-b, -a]]
    
    This creates a rotation + contraction system:
    - a > 0: contraction toward origin
    - b ≠ 0: rotation (counterclockwise if b > 0)
    
    Args:
        x: current positions (n_points, 2)
        t: current time
        a: contraction coefficient (controls decay)
        b: rotation coefficient (controls angular velocity)
    
    Returns:
        drift field (n_points, 2)
    '''
    if x.shape[1] != 2:
        raise ValueError("Linear drift with this A matrix only works in 2D")
    
    # Drift matrix A
    A = np.array([[-a,  b],
                  [-b, -a]])
    
    # Apply: f(x) = A·x for each point
    return x @ A.T  # (n_points, 2) @ (2, 2)^T = (n_points, 2)

# Linear attractive drift
def attractive_drift(x: np.ndarray, t: float, strength: float = 1, center: np.ndarray = None) -> np.ndarray:
    '''
    Drift function simulatin an attraction towards the center (0, 0) of the cartesian plain
    
    Args:
        x: current positions
        t: current time
        strength: attraction intensity

    Return:
        drift_field: in the shape of (n_points, dim), it points towards the origin
    '''
    if center is None:
        center = np.zeros(x.shape[1])

    return -strength * (x - center)

# Rotational drift 
def rotational_drift(x: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    Drift rotazionale (solo 2D)
    
    Formula: f(x, y) = [-ω*y, ω*x]
    
    Effetto: I punti ruotano attorno all'origine con velocità angolare ω
    
    Args:
        x: posizioni correnti (n_points, 2)
        t: tempo corrente
        omega: ω = velocità angolare (rad/s)
    
    Returns:
        drift field (n_points, 2)
    """
    if x.shape[1] != 2:
        raise ValueError("Rotational drift funziona solo in 2D")
    
    # Matrice di rotazione: [[0, -ω], [ω, 0]]
    rotation_matrix = np.array([[0, -omega], [omega, 0]])
    return x @ rotation_matrix.T


# 3. Combined drift (linear attractive + rotational) 
def combined_drift(x: np.ndarray, t: float, 
                  attractive_strength: float = 0.5,
                  rotational_omega: float = 0.5) -> np.ndarray:
    """
    Combinazione di drift attrattivo e rotazionale
    
    Formula: f(x) = -α*x + rotazione(x)
    
    Effetto: I punti spiraleggiano verso il centro
    
    Args:
        x: posizioni correnti (n_points, 2)
        t: tempo corrente
        attractive_strength: α = intensità attrazione
        rotational_omega: ω = velocità rotazione
    
    Returns:
        drift field (n_points, 2)
    """
    return (attractive_drift(x, t, attractive_strength) + 
            rotational_drift(x, t, rotational_omega))


# Double well drift
def double_well_drift(x: np.ndarray, t: float, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """
    Drift derivato da un potenziale a doppio pozzo
    
    Potenziale: V(x) = -a*x²/2 + b*x⁴/4
    Drift: f(x) = -∇V(x) = a*x - b*x³
    
    Effetto: Due punti di equilibrio stabile a x = ±√(a/b), 
             un punto instabile in x = 0
    
    Args:
        x: posizioni correnti (n_points, dim)
        t: tempo corrente
        a: parametro (profondità dei pozzi)
        b: parametro (separazione dei pozzi)
    
    Returns:
        drift field (n_points, dim)
    """
    return a * x - b * (x ** 3)


# Time-varying drift
def time_varying_drift(x: np.ndarray, t: float, T: float = 5.0) -> np.ndarray:
    """
    Drift attrattivo che aumenta nel tempo
    
    Formula: f(x, t) = -α(t) * x, dove α(t) = 0.1 + 0.9*(t/T)
    
    Effetto: Attrazione debole all'inizio, forte alla fine
    
    Args:
        x: posizioni correnti (n_points, dim)
        t: tempo corrente
        T: tempo finale
    
    Returns:
        drift field (n_points, dim)
    """
    strength = 0.1 + 0.9 * (t / T)
    return -strength * x


# Repulsive drift
def repulsive_drift(x: np.ndarray, t: float, strength: float = 1.0) -> np.ndarray:
    """
    Drift repulsivo dall'origine
    
    Formula: f(x) = +α * x
    
    Effetto: Spinge i punti lontano dal centro (esplosione)
    
    Args:
        x: posizioni correnti (n_points, dim)
        t: tempo corrente
        strength: α = intensità della repulsione
    
    Returns:
        drift field (n_points, dim)
    """
    return strength * x


# Gradient flow (Ornstein-Uhlenbeck)
def ornstein_uhlenbeck_drift(x: np.ndarray, t: float, 
                             theta: float = 1.0, 
                             mu: np.ndarray = None) -> np.ndarray:
    """
    Processo di Ornstein-Uhlenbeck (mean-reverting)
    
    Formula: f(x) = θ(μ - x)
    
    Effetto: I punti vengono attratti verso una media μ con velocità θ
             (usato in finanza per modellare tassi di interesse)
    
    Args:
        x: posizioni correnti (n_points, dim)
        t: tempo corrente
        theta: θ = velocità di mean-reversion
        mu: μ = punto di equilibrio (default: origine)
    
    Returns:
        drift field (n_points, dim)
    """
    if mu is None:
        mu = np.zeros(x.shape[1])
    return theta * (mu - x)


# Vortex drift
def vortex_drift(x: np.ndarray, t: float, strength: float = 1.0) -> np.ndarray:
    """
    Drift che crea un vortice (rotazione + attrazione radiale)
    
    Formula: componente tangenziale + componente radiale
    
    Effetto: I punti spiraleggiano verso il centro come in un vortice
    
    Args:
        x: posizioni correnti (n_points, 2)
        t: tempo corrente
        strength: intensità del vortice
    
    Returns:
        drift field (n_points, 2)
    """
    if x.shape[1] != 2:
        raise ValueError("Vortex drift funziona solo in 2D")
    
    # Distanza dall'origine
    r = np.linalg.norm(x, axis=1, keepdims=True)
    r = np.where(r < 1e-8, 1e-8, r)  # Evita divisione per zero
    
    # Componente tangenziale (rotazione)
    tangential = np.column_stack([-x[:, 1], x[:, 0]])
    
    # Componente radiale (attrazione)
    radial = -x / r
    
    return strength * (tangential + radial)


# Saddle drift
def saddle_drift(x: np.ndarray, t: float, 
                lambda_stable: float = 1.0,
                lambda_unstable: float = 1.0) -> np.ndarray:
    """
    Drift con punto di sella (2D)
    
    Formula: f(x, y) = [-λ_s*x, +λ_u*y]
    
    Effetto: Attrazione lungo x, repulsione lungo y (forma a sella)
    
    Args:
        x: posizioni correnti (n_points, 2)
        t: tempo corrente
        lambda_stable: λ_s = tasso attrazione (direzione x)
        lambda_unstable: λ_u = tasso repulsione (direzione y)
    
    Returns:
        drift field (n_points, 2)
    """
    if x.shape[1] != 2:
        raise ValueError("Saddle drift funziona solo in 2D")
    
    drift = np.zeros_like(x)
    drift[:, 0] = -lambda_stable * x[:, 0]   # Attrazione lungo x
    drift[:, 1] = lambda_unstable * x[:, 1]   # Repulsione lungo y
    return drift


# Langevin dynamics 
def langevin_drift(x: np.ndarray, t: float, 
                  potential_gradient: callable = None,
                  gamma: float = 1.0) -> np.ndarray:
    """
    Drift per dinamica di Langevin
    
    Formula: f(x) = -γ * ∇V(x)
    
    Effetto: I punti seguono il gradiente di un potenziale V(x)
    
    Args:
        x: posizioni correnti (n_points, dim)
        t: tempo corrente
        potential_gradient: funzione che calcola ∇V(x)
        gamma: γ = coefficiente di friction
    
    Returns:
        drift field (n_points, dim)
    """
    if potential_gradient is None:
        # Default: potenziale quadratico V(x) = x²/2, quindi ∇V(x) = x
        grad_V = x
    else:
        grad_V = potential_gradient(x, t)
    
    return -gamma * grad_V