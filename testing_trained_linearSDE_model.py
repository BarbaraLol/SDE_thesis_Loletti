"""
Debug script to test the ACTUAL trained model.
Run this after training completes (or at any checkpoint).

Usage:
  python debug.py --checkpoint experiments/linear_sde/checkpoints/checkpoint_1.pth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import argparse
import sys
import os

# Import your modules
import sde_lib
from models import utils as mutils
from configs import linear_sde_2d


def load_trained_model(checkpoint_path, config):
    """Load a trained score model from checkpoint"""
    score_model = mutils.create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # Handle DataParallel wrapped models
    state_dict = checkpoint['model']
    
    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    score_model.load_state_dict(new_state_dict)
    score_model.eval()
    score_model.to(config.device)
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Training step: {checkpoint['step']}")
    
    return score_model


def test_score_direction(sde, score_model, config):
    """
    Test if the learned score function points in reasonable directions.
    
    For a Gaussian centered at origin, the score should point toward the origin.
    We can't test exact accuracy without knowing the true x₀, but we can check
    if the direction makes qualitative sense.
    """
    print("\n" + "=" * 80)
    print("TESTING LEARNED SCORE FUNCTION DIRECTION")
    print("=" * 80)
    
    device = config.device
    
    # Test at different times
    test_times = [0.2, 0.5, 0.8]
    
    for t_val in test_times:
        print(f"\n--- Time t = {t_val} ---")
        
        # Create test points at different positions
        test_points = torch.tensor([
            [[2.0], [1.0]],    # Upper right
            [[-2.0], [-1.0]],  # Lower left
            [[1.0], [-1.0]],   # Lower right
            [[-1.0], [1.0]],   # Upper left
        ], dtype=torch.float32).view(4, 2, 1, 1).to(device)
        
        t = torch.full((4,), t_val, device=device)
        
        # Get score from trained model
        with torch.no_grad():
            scores = score_model(test_points, t)
        
        print("\nTest points and their scores:")
        print(f"{'Point':<20} {'Score':<20} {'Points toward origin?'}")
        print("-" * 60)
        
        for i in range(4):
            point = test_points[i].squeeze().cpu().numpy()
            score = scores[i].squeeze().cpu().numpy()
            
            # Check if score points generally toward origin
            # Score should be opposite sign to position for Gaussian centered at 0
            toward_origin = -point
            cos_angle = np.dot(toward_origin, score) / (
                np.linalg.norm(toward_origin) * np.linalg.norm(score) + 1e-8
            )
            
            points_correct = "✓ Yes" if cos_angle > 0 else "✗ No"
            
            print(f"[{point[0]:5.2f}, {point[1]:5.2f}]      "
                  f"[{score[0]:6.2f}, {score[1]:6.2f}]      "
                  f"{points_correct} (cos={cos_angle:.3f})")


def test_forward_backward_with_trained_model(sde, score_model, config):
    """
    Test forward-backward process using the TRAINED model.
    
    This doesn't expect perfect reconstruction (the model is approximate),
    but we can check if:
    1. Samples don't explode
    2. Reverse process moves samples toward reasonable regions
    3. Final samples are in the expected distribution
    """
    print("\n" + "=" * 80)
    print("FORWARD-BACKWARD WITH TRAINED MODEL")
    print("=" * 80)
    
    device = config.device
    n_samples = 10
    
    # Start from prior (noisy distribution)
    print("\nSampling from prior p_T...")
    x_T = sde.prior_sampling((n_samples, 2, 1, 1)).to(device)
    print(f"Prior samples (should be noisy):\n{x_T.squeeze().cpu().numpy()}")
    
    # Create score function from trained model
    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
    
    # Reverse process
    print("\nRunning reverse process (denoising)...")
    rsde = sde.reverse(score_fn, probability_flow=False)
    
    x_t = x_T.clone()
    dt = sde.T / sde.N
    
    # Store trajectory
    trajectory = [x_t.clone()]
    
    for step in range(sde.N):
        t = torch.ones(n_samples, device=device) * (sde.T - step * dt)
        
        drift, diffusion = rsde.sde(x_t, t)
        z = torch.randn_like(x_t)
        
        # Handle both scalar and tensor diffusion
        if isinstance(diffusion, torch.Tensor):
            if diffusion.dim() == 1:
                x_t = x_t + drift * dt + diffusion[:, None, None, None] * np.sqrt(dt) * z
            else:
                x_t = x_t + drift * dt + diffusion * np.sqrt(dt) * z
        else:
            # diffusion is 0 for probability flow
            x_t = x_t + drift * dt
        
        if step % (sde.N // 5) == 0:
            trajectory.append(x_t.clone())
            progress = (step + 1) / sde.N * 100
            print(f"  Progress: {progress:.1f}%")
    
    x_0 = x_t
    
    print(f"\nGenerated samples (after denoising):\n{x_0.squeeze().detach().cpu().numpy()}")
    
    # Check statistics
    samples_flat = x_0.squeeze().detach().cpu().numpy()
    mean = samples_flat.mean(axis=0)
    std = samples_flat.std(axis=0)
    
    print(f"\nStatistics of generated samples:")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    print(f"  Range: [{samples_flat.min():.2f}, {samples_flat.max():.2f}]")
    
    # Sanity checks
    print("\nSanity checks:")
    if np.abs(samples_flat).max() < 100:
        print("  ✓ Samples are bounded (not exploding)")
    else:
        print("  ✗ WARNING: Samples are very large (possible divergence)")
    
    if np.abs(mean).max() < 5:
        print("  ✓ Mean is near origin (as expected for toy data)")
    else:
        print("  ✗ WARNING: Mean is far from expected region")
    
    # Visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    for i in range(n_samples):
        # Plot trajectory
        traj_x = [trajectory[j][i, 0, 0, 0].cpu().item() for j in range(len(trajectory))]
        traj_y = [trajectory[j][i, 1, 0, 0].cpu().item() for j in range(len(trajectory))]
        ax.plot(traj_x, traj_y, c=colors[i], alpha=0.5, linewidth=1.5)
        
        # Mark start and end
        ax.scatter(traj_x[0], traj_y[0], c=[colors[i]], marker='s', 
                   s=100, alpha=0.5, edgecolors='black', linewidths=1)
        ax.scatter(traj_x[-1], traj_y[-1], c=[colors[i]], marker='o', 
                   s=100, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Reverse Process Trajectories\n(□=noisy start, ○=denoised end)', 
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('trained_model_samples.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'trained_model_samples.png'")
    plt.show()


def compare_learned_vs_exact_score(sde, score_model, config):
    """
    Compare the learned score function against the exact score.
    This requires generating test data and computing exact scores.
    """
    print("\n" + "=" * 80)
    print("COMPARING LEARNED VS EXACT SCORE")
    print("=" * 80)
    
    device = config.device
    n_samples = 50
    
    # Generate test data from the target distribution
    x0_test = torch.randn(n_samples, 2, 1, 1) * 2.0  # Gaussian samples
    
    # Test at different times
    test_times = [0.3, 0.6, 0.9]
    
    for t_val in test_times:
        print(f"\n--- Time t = {t_val} ---")
        
        t = torch.full((n_samples,), t_val, device=device)
        
        # Forward diffuse the test data
        mean_t, std_t = sde.marginal_prob(x0_test.to(device), t)
        x_t = mean_t + std_t * torch.randn_like(x0_test.to(device))
        
        # Get learned score
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
        with torch.no_grad():
            learned_scores = score_fn(x_t, t)
        
        # Compute exact score
        exact_scores = []
        for i in range(n_samples):
            # Compute Σ(t)
            exp_neg_At = expm(-sde.A_np * t_val)
            exp_neg_ATt = expm(-sde.A_np.T * t_val)
            Sigma_t = sde.Sigma_inf - exp_neg_At @ sde.Sigma_inf @ exp_neg_ATt
            
            # Add small regularization
            Sigma_t += np.eye(2) * 1e-6
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            
            # Score: -Σ^{-1}(x - μ)
            x_i = x_t[i].view(2).cpu().numpy()
            mean_i = mean_t[i].view(2).cpu().numpy()
            score_flat = -Sigma_t_inv @ (x_i - mean_i)
            exact_scores.append(torch.tensor(score_flat, dtype=torch.float32).view(2, 1, 1))
        
        exact_scores = torch.stack(exact_scores).to(device)
        
        # Compute error
        error = torch.abs(learned_scores - exact_scores)
        mse = torch.mean((learned_scores - exact_scores) ** 2)
        
        print(f"  MSE between learned and exact: {mse:.6f}")
        print(f"  Mean absolute error: {torch.mean(error):.6f}")
        print(f"  Max absolute error: {torch.max(error):.6f}")
        
        # Compute correlation
        learned_flat = learned_scores.view(n_samples, 2).cpu().numpy()
        exact_flat = exact_scores.view(n_samples, 2).cpu().numpy()
        
        corr_x1 = np.corrcoef(learned_flat[:, 0], exact_flat[:, 0])[0, 1]
        corr_x2 = np.corrcoef(learned_flat[:, 1], exact_flat[:, 1])[0, 1]
        
        print(f"  Correlation (x₁): {corr_x1:.4f}")
        print(f"  Correlation (x₂): {corr_x2:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    # Load config
    config = linear_sde_2d.get_config()
    
    # Create SDE
    sde = sde_lib.LinearSDE(
        A_matrix=config.model.A_matrix,
        epsilon=config.model.epsilon,
        N=config.model.num_scales
    )
    
    print(f"\nSDE Configuration:")
    print(f"  A matrix:\n{np.array(config.model.A_matrix)}")
    print(f"  ε = {config.model.epsilon}")
    print(f"  N = {config.model.num_scales}")
    
    # Load trained model
    score_model = load_trained_model(args.checkpoint, config)
    
    # Run tests
    test_score_direction(sde, score_model, config)
    test_forward_backward_with_trained_model(sde, score_model, config)
    compare_learned_vs_exact_score(sde, score_model, config)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\nFiles generated:")
    print("  - trained_model_samples.png")


if __name__ == "__main__":
    main()