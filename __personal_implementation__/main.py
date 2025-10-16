"""
Main script for individual time SDE approach
Each point has its own terminal time T_i
"""

import numpy as np
import torch
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

from Individual_Time_ScoreNet import IndividualTimeSDESolver, DenoisingScoreMatcher
from SDE_integrator import SDEconfig
from ScoreNet import DenoisingScoreMatcher
import manifolds
import drift_functions


def load_config(config_name: str):
    if config_name == 'linear_arc':
        from config.linear_arc_example import get_config
    elif config_name == 'arc':
        from config.arc_example import get_config
    elif config_name == 'disk':
        from config.disk_example import get_config
    else:
        raise ValueError(f"Config '{config_name}' not found")
    return get_config()


def create_drift_function(config):
    drift_type = config.drift.type
    
    if drift_type == 'linear':
        a = config.drift.a
        b = config.drift.b
        return lambda x, t: drift_functions.linear_drift(x, t, a=a, b=b)
    elif drift_type == 'attractive':
        strength = config.drift.strength
        return lambda x, t: drift_functions.attractive_drift(x, t, strength=strength)
    else:
        raise ValueError(f"Drift type '{drift_type}' not supported")


def create_initial_manifold(config):
    mtype = config.manifold.type
    n_points = config.data.n_points
    radius = config.manifold.radius
    
    if mtype == 'arc':
        return manifolds.create_arc_manifold(n_points, radius, config.manifold.arc_fraction)
    elif mtype == 'disk':
        return manifolds.create_disk_manifold(n_points, radius)
    else:
        raise ValueError(f"Manifold type '{mtype}' not supported")


def plot_terminal_time_distribution(terminal_times, output_dir):
    """Plot distribution of terminal times"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(terminal_times, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(terminal_times.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {terminal_times.mean():.2f}s')
    ax.set_xlabel('Terminal Time (s)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Distribution of Terminal Times', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'terminal_times_distribution.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_sample_trajectories(trajectories, time_arrays, terminal_times, x0, output_dir, n_samples=20):
    """Plot sample forward/backward trajectories"""
    n_samples = min(n_samples, len(trajectories))
    sample_indices = np.random.choice(len(trajectories), n_samples, replace=False)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis((terminal_times[sample_indices] - terminal_times.min()) / 
                            (terminal_times.max() - terminal_times.min()))
    
    for i, idx in enumerate(sample_indices):
        traj = trajectories[idx]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, 
                   marker='o', edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=100,
                   marker='s', edgecolors='black', linewidths=2, zorder=10)
    
    ax.scatter(0, 0, c='red', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Center', zorder=100)
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title(f'Sample Trajectories (n={n_samples}, colored by terminal time)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'sample_trajectories.png'),
               dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_reconstruction_analysis(x0, reconstructed, terminal_times, output_dir):
    """Plot reconstruction error analysis"""
    errors = np.array([np.linalg.norm(x0[i] - reconstructed[i]) 
                      for i in range(len(x0))])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error vs terminal time
    ax1 = axes[0]
    scatter = ax1.scatter(terminal_times, errors, c=errors, cmap='viridis', 
                       s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax1.axhline(errors.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {errors.mean():.4f}')
    ax1.set_xlabel('Terminal Time (s)', fontsize=13)
    ax1.set_ylabel('Reconstruction Error', fontsize=13)
    ax1.set_title('Reconstruction Error vs Terminal Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Error')
    
    # Reconstruction visualization
    ax2 = axes[1]
    ax2.scatter(x0[:, 0], x0[:, 1], c='green', s=60, alpha=0.7,
               label='Original (t=0)', zorder=10)
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], 
               c='blue', s=60, alpha=0.7, marker='x',
               label='Reconstructed', zorder=10)
    
    # Lines connecting original to reconstructed (subsample for clarity)
    for i in range(0, len(x0), max(1, len(x0)//50)):
        ax2.plot([x0[i, 0], reconstructed[i, 0]],
                [x0[i, 1], reconstructed[i, 1]],
                'gray', alpha=0.2, linewidth=0.5)
    
    ax2.scatter(0, 0, c='red', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Center', zorder=100)
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('y', fontsize=13)
    ax2.set_title(f'Reconstruction (Mean Error: {errors.mean():.4f})', 
                 fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'reconstruction_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Individual Time SDE')
    parser.add_argument('--config', type=str, default='linear_arc')
    parser.add_argument('--min_time_factor', type=float, default=0.5)
    parser.add_argument('--max_time_factor', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--subsample_training', type=int, default=1)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--do_reverse', action='store_true')
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  INDIVIDUAL TIME SDE")
    print(f"{'='*70}\n")
    
    # Load configuration
    config = load_config(args.config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/individual_times_{args.config}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}\n")
    
    # Create SDE config
    sde_config = SDEconfig(
        dim=config.data.dim,
        T=config.data.T,
        dt=config.data.dt,
        epsilon=config.data.epsilon
    )
    
    # Create initial manifold
    x0 = create_initial_manifold(config)
    print(f"Initial Manifold:")
    print(f"  • Type: {config.manifold.type.upper()}")
    print(f"  • Points: {config.data.n_points}")
    print(f"  • Radius: {config.manifold.radius}\n")
    
    # Create drift function
    drift_fn = create_drift_function(config)
    
    # Create solver
    solver = IndividualTimeSDESolver(sde_config, drift_fn)
    
    # ============= FORWARD SDE =============
    print(f"{'→'*35}")
    print(f"  FORWARD SDE (Individual Times)")
    print(f"{'→'*35}\n")
    
    min_time = args.min_time_factor * sde_config.T
    max_time = args.max_time_factor * sde_config.T
    
    forward_trajs, forward_times, terminal_times = solver.forward_individual_times(
        x0,
        min_time=min_time,
        max_time=max_time,
        save_every=args.save_every
    )
    
    # Save terminal times
    np.save(os.path.join(output_dir, 'terminal_times.npy'), terminal_times)
    
    # Get final positions
    final_positions = np.array([traj[-1] for traj in forward_trajs])
    
    # Plot terminal time distribution
    if not args.no_plot:
        plot_terminal_time_distribution(terminal_times, output_dir)
        plot_sample_trajectories(forward_trajs, forward_times, terminal_times, x0, output_dir)
    
    # ============= CREATE TRAINING DATASET =============
    print(f"{'📊'*35}")
    print(f"  CREATE TRAINING DATASET")
    print(f"{'📊'*35}\n")
    
    training_dataset = solver.create_training_dataset(
        forward_trajs, forward_times, subsample=args.subsample_training
    )
    
    # ============= TRAIN SCORE NETWORK =============
    print(f"{'🧠'*35}")
    print(f"  SCORE NETWORK TRAINING")
    print(f"{'🧠'*35}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert training dataset to trajectory format
    time_dict = {}
    for pos, t in training_dataset:
        t_key = f"{t:.6f}"
        if t_key not in time_dict:
            time_dict[t_key] = []
        time_dict[t_key].append(pos)
    
    # Convert to trajectory format
    traj_list = []
    time_list = []
    for t_key in sorted(time_dict.keys()):
        positions = np.array(time_dict[t_key])
        traj_list.append(positions)
        time_list.append(float(t_key))
    
    dsm = DenoisingScoreMatcher(
        forward_trajectory=traj_list,
        forward_times=np.array(time_list),
        dim=config.data.dim,
        device=device,
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers
    )
    
    if args.load_model:
        dsm.load_model(args.load_model)
    else:
        loss_history = dsm.train(
            n_epochs=config.training.n_epochs,
            batch_size=config.training.batch_size,
            lr=config.training.lr,
            sigma_dn=config.training.sigma_dn,
            weight_decay=getattr(config.training, 'weight_decay', 0.0),
            verbose=True
        )
        
        # Save model
        model_path = os.path.join(output_dir, 'checkpoints', 'trained_model.pt')
        dsm.save_model(model_path)
        
        # Plot loss
        if not args.no_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(loss_history, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=13)
            ax.set_ylabel('Loss', fontsize=13)
            ax.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plots', 'training_loss.png'),
                       dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    
    score_fn = dsm.get_score_function()
    
    # ============= BACKWARD SDE =============
    if args.do_reverse:
        print(f"{'←'*35}")
        print(f"  BACKWARD SDE (Individual Times)")
        print(f"{'←'*35}\n")
        
        backward_trajs, backward_times = solver.backward_individual_times(
            final_positions,
            terminal_times,
            score_fn,
            save_every=args.save_every
        )
        
        # ============= COMPUTE ERRORS =============
        print(f"{'='*70}")
        print(f"  RECONSTRUCTION RESULTS")
        print(f"{'='*70}\n")
        
        reconstructed_positions = np.array([traj[-1] for traj in backward_trajs])
        errors = np.array([np.linalg.norm(x0[i] - reconstructed_positions[i]) 
                          for i in range(len(x0))])
        
        mean_error = errors.mean()
        std_error = errors.std()
        
        print(f"Mean reconstruction error: {mean_error:.6f}")
        print(f"Std reconstruction error: {std_error:.6f}")
        print(f"Min error: {errors.min():.6f}")
        print(f"Max error: {errors.max():.6f}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Reconstruction Metrics (Individual Times)\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Mean reconstruction error: {mean_error:.6f}\n")
            f.write(f"Std reconstruction error: {std_error:.6f}\n")
            f.write(f"Min error: {errors.min():.6f}\n")
            f.write(f"Max error: {errors.max():.6f}\n")
            f.write(f"\nTerminal Times:\n")
            f.write(f"Mean: {terminal_times.mean():.2f}s\n")
            f.write(f"Min: {terminal_times.min():.2f}s\n")
            f.write(f"Max: {terminal_times.max():.2f}s\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Points: {config.data.n_points}\n")
            f.write(f"Epsilon: {config.data.epsilon}\n")
            f.write(f"dt: {config.data.dt}\n")
        
        print(f"\n💾 Metrics saved to: {metrics_path}")
        
        # Plot reconstruction analysis
        if not args.no_plot:
            plot_reconstruction_analysis(x0, reconstructed_positions, terminal_times, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ Simulation completed!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()