import numpy as np

def create_arc_manifold(n_points: int, radius: float, arc_fraction: float = 0.75) -> np.ndarray:
    """
    Creation of points in an arc of a circle with uniform distribution 
    
    Args:
        n_points: number of points
        radius: circle radius
        arc_fraction: circle fractions (es. 0.75 = 3/4 of the circle)
    
    Returns:
        Array (n_points, 2) with the points' coordinates
    """
    theta = np.linspace(0, 2 * np.pi * arc_fraction, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def create_disk_manifold(n_points: int, radius: float) -> np.ndarray:
    """
    Creation of points inside a disk with uniform distribution 
    
    Args:
        n_points: number of points
        radius: disk radius
    
    Returns:
        Array (n_points, 2) with the points' coordinates
    """
    # Campionamento uniforme in un disco usando la trasformata inversa
    # r ~ sqrt(U[0,1]) per avere densità uniforme
    r = np.sqrt(np.random.rand(n_points)) * radius
    theta = np.random.rand(n_points) * 2 * np.pi
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])