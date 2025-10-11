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
    plt.show()