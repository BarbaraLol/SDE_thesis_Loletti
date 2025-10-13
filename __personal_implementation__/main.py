import numpy as np
import argparse
from SDE_integrator import SDEconfig, SDESolver
import manifolds
import drift_functions
import visualization
# from score_estimator import GaussianKDEEstimator, AdaptiveBandwidthKDEScoreEstimator
from score_estimator import AdaptiveGaussianScoreEstimator, run_adaptive_backward

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
    """Factory per creare il manifold iniziale"""
    mtype = config.manifold.type
    n_points = config.data.n_points
    radius = config.manifold.radius
    
    if mtype == 'arc':
        arc_fraction = config.manifold.arc_fraction
        return manifolds.create_arc_manifold(n_points, radius, arc_fraction)
    elif mtype == 'disk':
        return manifolds.create_disk_manifold(n_points, radius)
    else:
        raise ValueError(f"Manifold type '{mtype}' non supportato")


def create_drift_function(config):
    """Factory per creare la funzione di drift"""
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
            x, t, attractive_strength=attractive_strength, rotational_omega=rotational_omega)
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
            x, t, lambda_stable=lambda_stable, lambda_unstable=lambda_unstable)
    elif drift_type == 'ornstein_uhlenbeck':
        theta = config.drift.theta
        return lambda x, t: drift_functions.ornstein_uhlenbeck_drift(x, t, theta=theta)
    else:
        raise ValueError(f"Drift type '{drift_type}' non supportato")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SDE Forward-Backward Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Argomenti principali
    parser.add_argument('--config', '-c', type=str, default='arc', choices=['arc', 'disk'], help='Configurazione')
    parser.add_argument('--drift', '-d', type=str, default=None, help='Tipo di drift')
    
    # Parametri SDE
    parser.add_argument('--epsilon', type=float, default=None, help='Rumore stocastico')
    parser.add_argument('--T', type=float, default=None, help='Tempo finale')
    parser.add_argument('--dt', type=float, default=None, help='Timestep')
    
    # Parametri manifold
    parser.add_argument('--n_points', type=int, default=None, help='Numero di punti')
    parser.add_argument('--radius', type=float, default=None, help='Raggio manifold')
    
    # Parametri drift
    parser.add_argument('--strength', type=float, default=None)
    parser.add_argument('--omega', type=float, default=None)
    parser.add_argument('--attractive_strength', type=float, default=None)
    parser.add_argument('--rotational_omega', type=float, default=None)
    parser.add_argument('--a', type=float, default=None)
    parser.add_argument('--b', type=float, default=None)
    parser.add_argument('--lambda_stable', type=float, default=None)
    parser.add_argument('--lambda_unstable', type=float, default=None)
    parser.add_argument('--theta', type=float, default=None)
    
    # Visualizzazione
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_plot', action='store_true', help='Disabilita plot')
    
    # Backward SDE
    parser.add_argument("--do_reverse", action="store_true", help="Run reverse SDE")
    parser.add_argument("--k_neighbors", type=int, default=None, help="K-NN for KDE")
    parser.add_argument("--bandwidth", type=float, default=None, help="KDE bandwidth")
    parser.add_argument("--adaptive_bandwidth", action="store_true", help="Use adaptive bandwidth")

    # Number of points for backward sampling distribution
    parser.add_argument("--adaptive_backward", action="store_true", help="Use adaptive backward (updates distribution at each step)")
    parser.add_argument("--n_samples_per_point", type=int, default=10, help="Number of Gaussian samples per point")
    parser.add_argument("--local_std", type=float, default=0.1, help="Std of local Gaussians")
    
    return parser.parse_args()


def override_config_with_args(config, args):
    """Override configuration with command line arguments"""
    if args.epsilon is not None:
        config.data.epsilon = args.epsilon
    if args.T is not None:
        config.data.T = args.T
    if args.dt is not None:
        config.data.dt = args.dt
    if args.n_points is not None:
        config.data.n_points = args.n_points
    if args.radius is not None:
        config.manifold.radius = args.radius
    if args.drift is not None:
        config.drift.type = args.drift
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
    if args.save_every is not None:
        config.visualization.save_every = args.save_every
    if args.seed is not None:
        config.seed = args.seed
    
    return config


def main():
    """Main function per simulazione SDE Forward + Backward"""
    
    # Parse argomenti
    args = parse_arguments()
    
    print(f"\n{'='*70}")
    print(f"  SDE FORWARD {'+ BACKWARD ' if args.do_reverse else ''}SIMULATION")
    print(f"  Configuration: {args.config.upper()}")
    print(f"{'='*70}\n")
    
    # Carica configurazione
    config = load_config(args.config)
    config = override_config_with_args(config, args)
    
    # Set random seed
    np.random.seed(config.seed)
    print(f"Random seed: {config.seed}")
    
    # Crea configurazione SDE
    sde_config = SDEconfig(
        dim=config.data.dim,
        T=config.data.T,
        dt=config.data.dt,
        epsilon=config.data.epsilon
    )
    
    print(f"\nSDE Configuration:")
    print(f"  • Dim: {sde_config.dim}D")
    print(f"  • T: {sde_config.T} s")
    print(f"  • dt: {sde_config.dt} s")
    print(f"  • n_steps: {sde_config.n_steps}")
    print(f"  • epsilon: {sde_config.epsilon}")
    
    # Crea manifold iniziale
    x0 = create_initial_manifold(config)
    print(f"\nInitial Manifold:")
    print(f"  • Type: {config.manifold.type.upper()}")
    print(f"  • Points: {config.data.n_points}")
    print(f"  • Radius: {config.manifold.radius}")
    if config.manifold.type == 'arc':
        print(f"  • Arc fraction: {config.manifold.arc_fraction}")
    
    # Crea drift function
    drift_type = config.drift.type
    drift_fn = create_drift_function(config)
    print(f"\nDrift: {drift_type.upper()}")
    
    # Crea solver
    solver = SDESolver(sde_config, drift_fn)
    
    # ============= FORWARD SDE =============
    print(f"\n{'→'*35}")
    print(f"  FORWARD SDE (t: 0 → T)")
    print(f"{'→'*35}")
    
    save_every = config.visualization.save_every
    forward_traj, forward_times = solver.forward_trajectory(x0, save_every=save_every)
    
    print(f"✓ Forward completed!")
    print(f"  • Snapshots: {len(forward_traj)}")
    print(f"  • Final time: {forward_times[-1]:.3f} s")
    
    # Plot forward
    if not args.no_plot:
        visualization.plot_forward_trajectories(
            forward_traj, forward_times,
            title=f"Forward SDE: {config.name} - {drift_type}"
        )
    
    # ============= BACKWARD SDE =============
    if args.do_reverse:
        if args.adaptive_backward:
            # from score_estimator import run_adaptive_backward
            run_adaptive_backward(config, drift_fn, x0, args)
        else:
            # Rimuovi tutto il resto (non ti serve più)
            pass
    
    print("\n✓ Simulation completed!\n")


if __name__ == "__main__":
    main()