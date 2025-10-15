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
    elif config_name == 'linear_arc':
        from config.linear_arc_example import get_config
    else:
        raise ValueError(f"Config '{config_name}' non trovata. Usa 'arc', 'disk', o 'linear_arc'")
    
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
    
    if drift_type == 'linear':
        # Linear drift: dx/dt = A·x with A = [[-a, b], [-b, -a]]
        a = config.drift.a
        b = config.drift.b
        return lambda x, t: drift_functions.linear_drift(x, t, a=a, b=b)
    elif drift_type == 'attractive':
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
    
    # Main arguments
    parser.add_argument('--config', '-c', type=str, default='arc', choices=['arc', 'disk', 'linear_arc'], help='Configuration')
    parser.add_argument('--drift', '-d', type=str, default=None, help='Drift type')
    
    # SDE parameters
    parser.add_argument('--epsilon', type=float, default=None, help='Stochastic noise')
    parser.add_argument('--T', type=float, default=None, help='Final time')
    parser.add_argument('--dt', type=float, default=None, help='Timestep')
    
    # Manifold parameters
    parser.add_argument('--n_points', type=int, default=None, help='Number of points')
    parser.add_argument('--radius', type=float, default=None, help='Manifold radius')
    
    # Drift parameters
    parser.add_argument('--strength', type=float, default=None)
    parser.add_argument('--omega', type=float, default=None)
    parser.add_argument('--a', type=float, default=None, help='Linear drift: contraction coeff')
    parser.add_argument('--b', type=float, default=None, help='Linear drift: rotation coeff')
    
    # Visualization
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_plot', action='store_true', help='Disable plots')
    
    # Forward SDE options
    parser.add_argument('--random_times', action='store_true', help='Sample forward trajectory at random times')
    parser.add_argument('--n_snapshots', type=int, default=None, help='Number of snapshots for random time sampling')
    
    # Model saving/loading
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load pretrained model')
    
    # Backward SDE
    parser.add_argument('--do_reverse', action='store_true', help='Run reverse SDE')
    
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
    if args.a is not None:
        config.drift.a = args.a
    if args.b is not None:
        config.drift.b = args.b
    if args.save_every is not None:
        config.visualization.save_every = args.save_every
    if args.seed is not None:
        config.seed = args.seed
    
    return config


def main():
    """Main function for SDE simulation with score model training"""
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\n{'='*70}")
    print(f"  SDE FORWARD {'+ BACKWARD ' if args.do_reverse else ''}SIMULATION")
    print(f"  Configuration: {args.config.upper()}")
    if args.random_times:
        print(f"  Mode: RANDOM TIME SAMPLING")
    print(f"{'='*70}\n")
    
    # Load configuration
    config = load_config(args.config)
    # config = override_config_with_args(config, args)
    
    # Set random seed
    np.random.seed(config.seed)
    print(f"Random seed: {config.seed}")
    
    # Create SDE configuration
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
    
    # Create initial manifold
    x0 = create_initial_manifold(config)
    print(f"\nInitial Manifold:")
    print(f"  • Type: {config.manifold.type.upper()}")
    print(f"  • Points: {config.data.n_points}")
    print(f"  • Radius: {config.manifold.radius}")
    if config.manifold.type == 'arc':
        print(f"  • Arc fraction: {config.manifold.arc_fraction}")
    
    # Create drift function
    drift_type = config.drift.type
    drift_fn = create_drift_function(config)
    print(f"\nDrift: {drift_type.upper()}")
    
    # Create solver
    solver = SDESolver(sde_config, drift_fn)
    
    # ============= FORWARD SDE =============
    print(f"\n{'→'*35}")
    print(f"  FORWARD SDE (t: 0 → T)")
    print(f"{'→'*35}")
    
    if args.random_times:
        # Use random time sampling
        n_snapshots = args.n_snapshots if args.n_snapshots else None
        forward_traj, forward_times = solver.forward_trajectory_random_times(
            x0, n_snapshots=n_snapshots
        )
    else:
        # Use regular uniform time sampling
        save_every = config.visualization.save_every
        forward_traj, forward_times = solver.forward_trajectory(x0, save_every=save_every)
    
    print(f"✓ Forward completed!")
    print(f"  • Snapshots: {len(forward_traj)}")
    print(f"  • Time range: [{forward_times[0]:.3f}, {forward_times[-1]:.3f}]")
    print(f"  • Final positions saved: {forward_traj[-1].shape}")
    
    # Save final positions for backward SDE
    x_T = forward_traj[-1]
    
    # Plot forward
    if not args.no_plot:
        visualization.plot_forward_trajectories(
            forward_traj, forward_times,
            title=f"Forward SDE: {config.name} - {drift_type}"
        )
    
    # ============= SCORE FUNCTION TRAINING =============
    score_fn = None
    
    print(f"\n{'🧠'*35}")
    print(f"  NEURAL NETWORK SCORE TRAINING")
    print(f"{'🧠'*35}")
    
    try:
        import torch
        from ScoreNet import DenoisingScoreMatcher
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {device}")
        
        # Create denoising score matcher
        dsm = DenoisingScoreMatcher(
            forward_trajectory=forward_traj,
            forward_times=forward_times,
            dim=config.data.dim,
            device=device,
            hidden_dim=config.model.hidden_dim,
            n_layers=config.model.n_layers
        )
        
        # Load or train model
        if args.load_model:
            print(f"\nLoading pretrained model from {args.load_model}")
            dsm.load_model(args.load_model)
        else:
            print(f"\nTraining neural network...")
            print(f"  • Epochs: {config.training.n_epochs}")
            print(f"  • Batch size: {config.training.batch_size}")
            print(f"  • Learning rate: {config.training.lr}")
            print(f"  • Denoising σ: {config.training.sigma_dn}")
            
            loss_history = dsm.train(
                n_epochs=config.training.n_epochs,
                batch_size=config.training.batch_size,
                lr=config.training.lr,
                sigma_dn=config.training.sigma_dn,
                verbose=True
            )
            
            # Plot training loss
            if not args.no_plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history, linewidth=2)
                plt.xlabel('Epoch', fontsize=13)
                plt.ylabel('Loss', fontsize=13)
                plt.title('Training Loss', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.show()
        
        # Save model 
        save_path = args.save_model if args.save_model else getattr(config.model, 'save_path', None)
        if save_path:
            import os
            dirpath = os.path.dirname(save_path)
            if dirpath:  # avoid os.makedirs('') if user gives just a filename
                os.makedirs(dirpath, exist_ok=True)
            dsm.save_model(save_path)
            print(f"\n✓ Model saved to: {save_path}")
        else:
            print("\nℹNo save path provided (use --save_model or set config.model.save_path).")

        
        # Get score function
        score_fn = dsm.get_score_function()
        print(f"\n✓ Neural network score function ready!")
        
    except ImportError:
        print("\n Error: PyTorch not available.")
        print("Please install PyTorch: pip install torch")
        print("\nCannot proceed without score function.")
        return
    
    # ============= BACKWARD SDE =============
    if args.do_reverse:
        print(f"\n{'←'*35}")
        print(f"  BACKWARD SDE (t: T → 0)")
        print(f"{'←'*35}")
        
        save_every = config.visualization.save_every
        
        print(f"\nRunning backward SDE...")
        print(f"  • Starting from {x_T.shape[0]} points at t={forward_times[-1]:.2f}")
        
        backward_traj, backward_times = solver.backward_trajectory(
            x_T, score_fn, save_every=save_every
        )
        
        print(f"✓ Backward completed!")
        print(f"  • Snapshots: {len(backward_traj)}")
        print(f"  • Final time: {backward_times[-1]:.3f} s")
        
        # Compute reconstruction error
        x_reconstructed = backward_traj[-1]
        reconstruction_error = np.mean(np.linalg.norm(x0 - x_reconstructed, axis=1))
        
        print(f"\n{'='*70}")
        print(f"  RECONSTRUCTION RESULTS")
        print(f"{'='*70}")
        print(f"  • Mean reconstruction error: {reconstruction_error:.4f}")
        print(f"  • Initial spread: {np.mean(np.linalg.norm(x0, axis=1)):.4f}")
        print(f"  • Final spread: {np.mean(np.linalg.norm(x_reconstructed, axis=1)):.4f}")
        
        # Plot comparison
        if not args.no_plot:
            visualization.plot_forward_backward_comparison(
                forward_traj, forward_times,
                backward_traj, backward_times,
                x0,
                title=f"Forward-Backward SDE: {config.name}"
            )
    
    print("\n✓ Simulation completed!\n")


if __name__ == "__main__":
    main()