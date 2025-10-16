import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_forward_trajectories(trajectory: List[np.ndarray], times: np.ndarray, 
                              title: str = "Forward SDE Evolution",
                              figsize: tuple = (16, 6)):
    """
    Visualizza l'evoluzione forward della SDE
    
    Args:
        trajectory: lista di array (n_points, dim) per ogni timestep salvato
        times: array dei tempi corrispondenti
        title: titolo del plot
        figsize: dimensione della figura
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    trajectory_array = np.array(trajectory)  # (n_timesteps, n_points, dim)
    n_timesteps, n_points, dim = trajectory_array.shape
    
    # ============= Plot 1: Snapshot Temporali =============
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0, 1, n_timesteps))
    
    # Mostra solo alcuni snapshot per chiarezza
    indices_to_show = np.linspace(0, n_timesteps-1, min(8, n_timesteps), dtype=int)
    
    for i in indices_to_show:
        positions = trajectory_array[i]
        ax1.scatter(positions[:, 0], positions[:, 1], 
                   c=[colors[i]], s=40, alpha=0.7, 
                   label=f't={times[i]:.2f}s')
    
    # Centro attrattivo
    ax1.scatter(0, 0, c='red', s=300, marker='*', 
               edgecolors='black', linewidths=2.5, 
               label='Centro Attrattivo', zorder=100)
    
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
    ax1.set_title('Snapshot Temporali', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    
    # ============= Plot 2: Traiettorie Individuali =============
    ax2 = axes[1]
    
    # Plotta solo alcune traiettorie per chiarezza
    n_traj_to_plot = min(20, n_points)
    indices_to_plot = np.linspace(0, n_points-1, n_traj_to_plot, dtype=int)
    
    for idx in indices_to_plot:
        traj = trajectory_array[:, idx, :]  # (n_timesteps, 2)
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=1.2, color='steelblue')
    
    # Posizioni iniziali e finali
    ax2.scatter(trajectory_array[0, :, 0], trajectory_array[0, :, 1], 
               c='blue', s=60, label='Posizioni Iniziali (t=0)', 
               zorder=10, edgecolors='black', linewidths=0.5)
    ax2.scatter(trajectory_array[-1, :, 0], trajectory_array[-1, :, 1], 
               c='green', s=60, label=f'Posizioni Finali (t={times[-1]:.1f})', 
               zorder=10, edgecolors='black', linewidths=0.5)
    
    # Centro attrattivo
    ax2.scatter(0, 0, c='red', s=300, marker='*', 
               edgecolors='black', linewidths=2.5, 
               label='Centro', zorder=100)
    
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('y', fontsize=13)
    ax2.set_title('Traiettorie Individuali', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axis('equal')
    ax2.legend(loc='best', fontsize=10)
    
    # ============= Plot 3: Distanza dal Centro nel Tempo =============
    ax3 = axes[2]
    
    # Calcola distanza media dal centro per ogni timestep
    distances_mean = np.array([np.mean(np.linalg.norm(pos, axis=1)) 
                               for pos in trajectory_array])
    distances_std = np.array([np.std(np.linalg.norm(pos, axis=1)) 
                              for pos in trajectory_array])
    
    # Plot con banda di confidenza
    ax3.plot(times, distances_mean, linewidth=2.5, color='darkblue', 
            label='Distanza Media')
    ax3.fill_between(times, 
                     distances_mean - distances_std, 
                     distances_mean + distances_std,
                     alpha=0.3, color='skyblue', label='±1 std')
    
    # Alcune traiettorie individuali
    for idx in indices_to_plot[:5]:
        traj_distances = np.linalg.norm(trajectory_array[:, idx, :], axis=1)
        ax3.plot(times, traj_distances, alpha=0.3, linewidth=1, 
                color='gray', linestyle='--')
    
    ax3.set_xlabel('Tempo (s)', fontsize=13)
    ax3.set_ylabel('Distanza dal Centro', fontsize=13)
    ax3.set_title('Evoluzione Distanze', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_forward_backward_comparison(forward_traj, forward_times,
                                     backward_traj, backward_times,
                                     x0, title="Forward-Backward SDE"):
    """
    Plot comparison between forward and backward trajectories
    
    Args:
        forward_traj: Forward trajectory snapshots
        forward_times: Forward trajectory times
        backward_traj: Backward trajectory snapshots
        backward_times: Backward trajectory times
        x0: Initial positions
        title: Plot title
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    forward_array = np.array(forward_traj)
    backward_array = np.array(backward_traj)
    
    # Plot 1: Forward Evolution
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(forward_traj)))
    
    for i in range(0, len(forward_traj), max(1, len(forward_traj)//8)):
        ax1.scatter(forward_traj[i][:, 0], forward_traj[i][:, 1], 
                   c=[colors[i]], s=30, alpha=0.6)
    
    ax1.scatter(x0[:, 0], x0[:, 1], c='green', s=80, marker='o',
               edgecolors='black', linewidths=1.5, label='Start (t=0)', zorder=100)
    ax1.scatter(forward_array[-1, :, 0], forward_array[-1, :, 1], 
               c='red', s=80, marker='s', edgecolors='black', linewidths=1.5,
               label=f'End (t=T)', zorder=100)
    ax1.scatter(0, 0, c='gold', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Target', zorder=50)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Forward SDE (t: 0→T)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Backward Evolution
    ax2 = axes[1]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(backward_traj)))
    
    for i in range(0, len(backward_traj), max(1, len(backward_traj)//8)):
        ax2.scatter(backward_traj[i][:, 0], backward_traj[i][:, 1], 
                   c=[colors[i]], s=30, alpha=0.6)
    
    ax2.scatter(backward_array[0, :, 0], backward_array[0, :, 1], 
               c='red', s=80, marker='s', edgecolors='black', linewidths=1.5,
               label='Start (t=T)', zorder=100)
    ax2.scatter(backward_array[-1, :, 0], backward_array[-1, :, 1], 
               c='green', s=80, marker='o', edgecolors='black', linewidths=1.5,
               label='End (t=0)', zorder=100)
    ax2.scatter(0, 0, c='gold', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Target', zorder=50)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Backward SDE (t: T→0)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Reconstruction
    ax3 = axes[2]
    ax3.scatter(x0[:, 0], x0[:, 1], c='green', s=60, alpha=0.7,
               label='Original (t=0)', zorder=10)
    ax3.scatter(backward_array[-1, :, 0], backward_array[-1, :, 1], 
               c='blue', s=60, alpha=0.7, marker='x',
               label='Reconstructed', zorder=10)
    
    # Lines connecting original to reconstructed
    for i in range(min(20, x0.shape[0])):
        ax3.plot([x0[i, 0], backward_array[-1, i, 0]],
                [x0[i, 1], backward_array[-1, i, 1]],
                'gray', alpha=0.3, linewidth=0.5)
    
    ax3.scatter(0, 0, c='gold', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Target', zorder=50)
    
    error = np.mean(np.linalg.norm(x0 - backward_array[-1], axis=1))
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title(f'Reconstruction (Error: {error:.3f})', 
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig