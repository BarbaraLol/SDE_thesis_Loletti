import numpy as np
import argparse
from SDE_integrator import SDEconfig, SDESolver
import manifolds
import drift_functions
import visualization


def load_config(config_name: str):
    """Carica configurazione da file usando ml_collections"""
    if config_name == 'arc':
        from config.arc_example import get_config
    elif config_name == 'disk':
        from config.disk_example import get_config
    else:
        raise ValueError(f"Config '{config_name}' non trovata. Usa 'arc' o 'disk'")
    
    return get_config()


def create_initial_manifold(config) -> np.ndarray:
    """
    Factory per creare il manifold iniziale
    
    Args:
        config: ml_collections.ConfigDict
    
    Returns:
        Array (n_points, dim) con posizioni iniziali
    """
    mtype = config.manifold.type
    n_points = config.data.n_points
    radius = config.manifold.radius
    
    if mtype == 'arc':
        arc_fraction = config.manifold.arc_fraction
        return manifolds.create_arc_manifold(n_points, radius, arc_fraction)
    
    elif mtype == 'disk':
        return manifolds.create_disk_manifold(n_points, radius)
    
    else:
        raise ValueError(f"Manifold type '{mtype}' non supportato. Usa 'arc' o 'disk'")


def create_drift_function(config):
    """
    Factory per creare la funzione di drift
    
    Args:
        config: ml_collections.ConfigDict
    
    Returns:
        Funzione drift(x, t) -> np.ndarray
    """
    drift_type = config.drift.type
    
    if drift_type == 'attractive':
        strength = config.drift.strength
        return lambda x, t: drift_functions.attractive_drift(x, t, strength=strength)
    
    elif drift_type == 'rotational':
        omega = config.drift.omega
        return lambda x, t: drift_functions.rotational_drift(x, t, omega=omega)
    
    elif drift_type == 'combined':
        attractive_strength = config.drift.attractive_strength
        rotational_omega = config.drift.rotational_omega
        return lambda x, t: drift_functions.combined_drift(
            x, t, 
            attractive_strength=attractive_strength,
            rotational_omega=rotational_omega
        )
    
    elif drift_type == 'double_well':
        a = config.drift.a
        b = config.drift.b
        return lambda x, t: drift_functions.double_well_drift(x, t, a=a, b=b)
    
    elif drift_type == 'time_varying':
        T = config.data.T
        return lambda x, t: drift_functions.time_varying_drift(x, t, T=T)
    
    elif drift_type == 'repulsive':
        strength = config.drift.strength
        return lambda x, t: drift_functions.repulsive_drift(x, t, strength=strength)
    
    elif drift_type == 'vortex':
        strength = config.drift.strength
        return lambda x, t: drift_functions.vortex_drift(x, t, strength=strength)
    
    elif drift_type == 'saddle':
        lambda_stable = config.drift.lambda_stable
        lambda_unstable = config.drift.lambda_unstable
        return lambda x, t: drift_functions.saddle_drift(
            x, t, 
            lambda_stable=lambda_stable,
            lambda_unstable=lambda_unstable
        )
    
    elif drift_type == 'ornstein_uhlenbeck':
        theta = config.drift.theta
        return lambda x, t: drift_functions.ornstein_uhlenbeck_drift(x, t, theta=theta)
    
    else:
        raise ValueError(f"Drift type '{drift_type}' non supportato")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SDE Forward Simulation Framework with ml_collections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  
  # Esempio base (usa config da file)
  python main.py --config arc
  python main.py --config disk
  
  # Override drift type
  python main.py --config arc --drift rotational --omega 0.8
  python main.py --config disk --drift combined --attractive_strength 0.5 --rotational_omega 0.6
  
  # Override parametri SDE
  python main.py --config arc --epsilon 0.2 --T 10.0
  
  # Cambia numero di punti
  python main.py --config disk --n_points 150
  
  # Combinazione completa
  python main.py --config arc --drift combined --attractive_strength 0.4 \\
                 --rotational_omega 0.8 --epsilon 0.1 --n_points 80

Drift disponibili:
  - attractive, rotational, combined, vortex, double_well
  - time_varying, repulsive, saddle, ornstein_uhlenbeck
        """
    )
    
    # Argomenti principali
    parser.add_argument('--config', '-c', type=str, default='arc', choices=['arc', 'disk'], help='Configurazione da caricare (default: arc)')
    
    parser.add_argument('--drift', '-d', type=str, default=None, help='Tipo di drift (override config file)')
    
    # Parametri SDE
    parser.add_argument('--epsilon', type=float, default=None, help='Intensità rumore stocastico')
    
    parser.add_argument('--T', type=float, default=None, help='Tempo finale simulazione')
    
    parser.add_argument('--dt', type=float, default=None, help='Timestep')
    
    # Parametri manifold
    parser.add_argument('--n_points', type=int, default=None, help='Numero di punti iniziali')
    
    parser.add_argument('--radius', type=float, default=None, help='Raggio manifold iniziale')
    
    # Parametri drift: attractive/repulsive/vortex
    parser.add_argument('--strength', type=float, default=None, help='Intensità drift')
    
    # Parametri drift: rotational
    parser.add_argument('--omega', type=float, default=None, help='Velocità angolare')
    
    # Parametri drift: combined
    parser.add_argument('--attractive_strength', type=float, default=None, help='Componente attrattiva (combined)')
    
    parser.add_argument('--rotational_omega', type=float, default=None, help='Componente rotazionale (combined)')
    
    # Parametri drift: double_well
    parser.add_argument('--a', type=float, default=None, help='Parametro a (double_well)')
    
    parser.add_argument('--b', type=float, default=None, help='Parametro b (double_well)')
    
    # Parametri drift: saddle
    parser.add_argument('--lambda_stable', type=float, default=None, help='Lambda stabile (saddle)')
    
    parser.add_argument('--lambda_unstable', type=float, default=None, help='Lambda instabile (saddle)')
    
    # Parametri drift: ornstein_uhlenbeck
    parser.add_argument('--theta', type=float, default=None, help='Theta (ornstein_uhlenbeck)')
    
    # Visualizzazione
    parser.add_argument('--save_every', type=int, default=None, help='Salva ogni N step')
    
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    parser.add_argument('--no_plot', action='store_true', help='Non mostrare i plot')
    
    return parser.parse_args()


def override_config_with_args(config, args):
    """
    Override della configurazione con argomenti da command line
    
    Args:
        config: ml_collections.ConfigDict
        args: argomenti parsati da argparse
    
    Returns:
        config aggiornata
    """
    # Override parametri SDE
    if args.epsilon is not None:
        config.data.epsilon = args.epsilon
    if args.T is not None:
        config.data.T = args.T
    if args.dt is not None:
        config.data.dt = args.dt
    
    # Override parametri manifold
    if args.n_points is not None:
        config.data.n_points = args.n_points
    if args.radius is not None:
        config.manifold.radius = args.radius
    
    # Override drift type
    if args.drift is not None:
        config.drift.type = args.drift
    
    # Override parametri drift specifici
    if args.strength is not None:
        config.drift.strength = args.strength
    if args.omega is not None:
        config.drift.omega = args.omega
    if args.attractive_strength is not None:
        config.drift.attractive_strength = args.attractive_strength
    if args.rotational_omega is not None:
        config.drift.rotational_omega = args.rotational_omega
    if args.a is not None:
        config.drift.a = args.a
    if args.b is not None:
        config.drift.b = args.b
    if args.lambda_stable is not None:
        config.drift.lambda_stable = args.lambda_stable
    if args.lambda_unstable is not None:
        config.drift.lambda_unstable = args.lambda_unstable
    if args.theta is not None:
        config.drift.theta = args.theta
    
    # Override visualizzazione
    if args.save_every is not None:
        config.visualization.save_every = args.save_every
    
    # Override seed
    if args.seed is not None:
        config.seed = args.seed
    
    return config


def main():
    """Main function per eseguire la simulazione SDE"""
    
    # Parse argomenti
    args = parse_arguments()
    
    # Carica configurazione
    config = load_config(args.config)
    
    # Override con argomenti command line
    config = override_config_with_args(config, args)
    
    # Set random seed per riproducibilità
    np.random.seed(config.seed)
    
    # Crea configurazione SDE
    sde_config = SDEconfig(
        dim=config.data.dim,
        T=config.data.T,
        dt=config.data.dt,
        epsilon=config.data.epsilon
    )
    
    # Crea manifold iniziale
    
    x0 = create_initial_manifold(config)
    
    # Crea funzione di drift
    drift_type = config.drift.type
    drift_fn = create_drift_function(config)
    
    # Crea solver
    solver = SDESolver(sde_config, drift_fn)
    
    save_every = config.visualization.save_every
    trajectory, times = solver.forward_trajectory(x0, save_every=save_every)
    
    # Visualizzazione e statistiche
    # visualization.print_statistics(trajectory, times, config.name)
    
    if not args.no_plot:
        visualization.plot_forward_trajectories(
            trajectory, times, 
            title=f"Forward SDE: {config.name} - Drift: {drift_type}"
        )
    else:
        print("\n(Plot disabilitati con --no_plot)")


if __name__ == "__main__":
    main()
