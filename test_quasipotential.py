"""test_quasipotential.py - Verify learned quasipotentials against paper results
Now saves plots into a folder (use --outdir, default 'plots')."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Import your config
from configs import quasipotential_ex1, quasipotential_ex2
from models import utils as mutils
from dynamical_quasipotential_system import Example1System, Example2System

import torch.nn as nn

def _unwrap(model):
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

def _model_device(model):
    return next(_unwrap(model).parameters()).device

def _ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def _maybe_add_module_prefix(state_dict):
    return {f"module.{k}" if not k.startswith("module.") else k: v
            for k, v in state_dict.items()}

def _maybe_remove_module_prefix(state_dict):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}

def load_trained_model(config, checkpoint_path):
    """Load a trained model from checkpoint, fixing DataParallel key mismatches."""
    model = mutils.create_model(config)

    # Load checkpoint (prefer safe/weights-only if available)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)  # PyTorch >= 2.1
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=config.device)  # fallback for older PyTorch

    state_dict = checkpoint.get("model", checkpoint)  # support both {"model": ...} and raw sd

    # Detect whether the *model* expects "module." keys
    model_keys = list(model.state_dict().keys())
    model_expects_module = model_keys and model_keys[0].startswith("module.")
    sd_has_module = list(state_dict.keys())[0].startswith("module.")

    # Harmonize prefixes
    if model_expects_module and not sd_has_module:
        state_dict = _maybe_add_module_prefix(state_dict)
    elif not model_expects_module and sd_has_module:
        state_dict = _maybe_remove_module_prefix(state_dict)

    # Load strictly; if you have BN running stats or other buffers differing, relax to strict=False
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def compute_quasipotential(model, x_points):
    """Compute U(x) = 2V(x) from the learned potential"""
    base = _unwrap(model)
    device = _model_device(model)

    with torch.no_grad():
        # Ensure input has correct shape and device/dtype
        if len(x_points.shape) == 2:
            x_points = x_points.unsqueeze(-1).unsqueeze(-1)

        x_in = x_points.squeeze(-1).squeeze(-1).to(device=device, dtype=next(base.parameters()).dtype)

        # Compute V(x) on the underlying module
        V_values = base.compute_V(x_in)

        # Quasipotential U = 2V (up to additive constant)
        U_values = 2 * V_values

    return U_values.detach().cpu().numpy().flatten()



def test_example1(checkpoint_path, config, outdir: str):
    """Test Example 1: 3D system with two equilibria"""
    print("\n" + "=" * 60)
    print("TESTING EXAMPLE 1: 3D System")
    print("=" * 60)

    # Load model
    model = load_trained_model(config, checkpoint_path)
    system = Example1System()

    # Define evaluation grid (paper uses domain [-2,2]^3)
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)

    # 1. Test along y=z=0 line (Figure 1, right panel in paper)
    print("\n1. Testing along y=z=0 line:")
    x_line = torch.tensor([[x, 0.0, 0.0] for x in x_range], dtype=torch.float32)
    U_learned = compute_quasipotential(model, x_line)
    U_true = np.array([system.true_quasipotential(np.array([x, 0, 0])) for x in x_range])

    # Shift to match at minimum
    U_learned = U_learned - U_learned.min()
    U_true = U_true - U_true.min()

    # Compute errors
    rmse = np.sqrt(np.mean((U_learned - U_true) ** 2))
    mae = np.mean(np.abs(U_learned - U_true))
    print(f"   RMSE along y=z=0: {rmse:.6f}")
    print(f"   MAE along y=z=0:  {mae:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Along y=z=0
    axes[0].plot(x_range, U_true, 'r-', label='True U', linewidth=2)
    axes[0].plot(x_range, U_learned, 'b--', label='Learned U', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('U(x,0,0)')
    axes[0].set_title('Quasipotential along y=z=0')
    axes[0].legend()
    axes[0].grid(True)

    # 2. Contour plot at z=0 (Figure 1, left panel in paper)
    print("\n2. Computing contour plot at z=0:")
    X, Y = np.meshgrid(x_range, y_range)
    Z_true = np.zeros_like(X)
    Z_learned = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([X[j, i], Y[j, i], 0.0])
            Z_true[j, i] = system.true_quasipotential(point)

            point_tensor = torch.tensor([[X[j, i], Y[j, i], 0.0]], dtype=torch.float32)
            Z_learned[j, i] = compute_quasipotential(model, point_tensor)[0]

    # Normalize both
    Z_true = Z_true - Z_true.min()
    Z_learned = Z_learned - Z_learned.min()

    # Compute 2D error metrics
    rmse_2d = np.sqrt(np.mean((Z_learned - Z_true) ** 2))
    mae_2d = np.mean(np.abs(Z_learned - Z_true))
    print(f"   RMSE on z=0 plane: {rmse_2d:.6f}")
    print(f"   MAE on z=0 plane:  {mae_2d:.6f}")

    # Plot true quasipotential
    contour_true = axes[1].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[1].plot([-1, 1], [0, 0], 'r*', markersize=15, label='Equilibria')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('True U(x,y,0)')
    plt.colorbar(contour_true, ax=axes[1])

    # Plot learned quasipotential
    contour_learned = axes[2].contourf(X, Y, Z_learned, levels=20, cmap='viridis')
    axes[2].plot([-1, 1], [0, 0], 'r*', markersize=15, label='Equilibria')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Learned U(x,y,0)')
    plt.colorbar(contour_learned, ax=axes[2])

    plt.tight_layout()
    out_path = os.path.join(outdir, 'example1_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   Saved comparison plot to '{out_path}'")

    # 3. Check equilibria
    print("\n3. Checking equilibria:")
    equilibria = [(-1, 0, 0), (1, 0, 0), (0, 0, 0)]
    eq_names = ['xa (stable)', 'xb (stable)', 'xc (saddle)']
    for eq, name in zip(equilibria, eq_names):
        point = torch.tensor([list(eq)], dtype=torch.float32)
        U_val = compute_quasipotential(model, point)[0]
        U_true_val = system.true_quasipotential(np.array(eq))
        print(f"   {name:15} : U_learned={U_val:.4f}, U_true={U_true_val:.4f}")

    return rmse, mae


def test_example2(checkpoint_path, config, outdir: str):
    """Test Example 2: 2D system with limit cycle"""
    print("\n" + "=" * 60)
    print("TESTING EXAMPLE 2: 2D System with Limit Cycle")
    print("=" * 60)

    # Load model
    model = load_trained_model(config, checkpoint_path)
    system = Example2System(a=1.0, b=2.5)

    # Define evaluation grid (paper uses domain [-0.5,2.5] x [1.0,4.0])
    x_range = np.linspace(-0.5, 2.5, 100)
    y_range = np.linspace(1.0, 4.0, 100)

    # 1. Test along y=b line (Figure 2, right panel in paper)
    print("\n1. Testing along y=2.5 line:")
    x_line = torch.tensor([[x, 2.5] for x in x_range], dtype=torch.float32)
    U_learned = compute_quasipotential(model, x_line)
    U_true = np.array([system.true_quasipotential(np.array([x, 2.5])) for x in x_range])

    # Normalize
    U_learned = U_learned - U_learned.min()
    U_true = U_true - U_true.min()

    # Compute errors
    rmse = np.sqrt(np.mean((U_learned - U_true) ** 2))
    mae = np.mean(np.abs(U_learned - U_true))
    print(f"   RMSE along y=2.5: {rmse:.6f}")
    print(f"   MAE along y=2.5:  {mae:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Along y=2.5
    axes[0].plot(x_range, U_true, 'r-', label='True U', linewidth=2)
    axes[0].plot(x_range, U_learned, 'b--', label='Learned U', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('U(x,2.5)')
    axes[0].set_title('Quasipotential along y=2.5')
    axes[0].legend()
    axes[0].grid(True)

    # 2. Contour plot (Figure 2, left panel in paper)
    print("\n2. Computing full 2D quasipotential:")
    X, Y = np.meshgrid(x_range, y_range)
    Z_true = np.zeros_like(X)
    Z_learned = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([X[j, i], Y[j, i]])
            Z_true[j, i] = system.true_quasipotential(point)

            point_tensor = torch.tensor([[X[j, i], Y[j, i]]], dtype=torch.float32)
            Z_learned[j, i] = compute_quasipotential(model, point_tensor)[0]

    # Normalize
    Z_true = Z_true - Z_true.min()
    Z_learned = Z_learned - Z_learned.min()

    # Compute 2D error metrics
    rmse_2d = np.sqrt(np.mean((Z_learned - Z_true) ** 2))
    mae_2d = np.mean(np.abs(Z_learned - Z_true))
    print(f"   RMSE on full domain: {rmse_2d:.6f}")
    print(f"   MAE on full domain:  {mae_2d:.6f}")

    # Plot true quasipotential
    contour_true = axes[1].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[1].plot(1.0, 2.5, 'r*', markersize=15, label='Unstable equilibrium')
    # Plot limit cycle
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.sqrt(0.5)
    x_cycle = 1.0 + r * np.cos(theta)
    y_cycle = 2.5 + r * np.sin(theta)
    axes[1].plot(x_cycle, y_cycle, 'w--', linewidth=2, label='Limit cycle')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('True U(x,y)')
    axes[1].legend()
    plt.colorbar(contour_true, ax=axes[1])

    # Plot learned quasipotential
    contour_learned = axes[2].contourf(X, Y, Z_learned, levels=20, cmap='viridis')
    axes[2].plot(1.0, 2.5, 'r*', markersize=15, label='Unstable equilibrium')
    axes[2].plot(x_cycle, y_cycle, 'w--', linewidth=2, label='Limit cycle')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Learned U(x,y)')
    axes[2].legend()
    plt.colorbar(contour_learned, ax=axes[2])

    plt.tight_layout()
    out_path = os.path.join(outdir, 'example2_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   Saved comparison plot to '{out_path}'")

    # 3. Check key points
    print("\n3. Checking key points:")
    center = (1.0, 2.5)
    point = torch.tensor([list(center)], dtype=torch.float32)
    U_val = compute_quasipotential(model, point)[0]
    U_true_val = system.true_quasipotential(np.array(center))
    print(f"   Center (1.0, 2.5): U_learned={U_val:.4f}, U_true={U_true_val:.4f}")

    # Point on limit cycle
    r = np.sqrt(0.5)
    cycle_point = (1.0 + r, 2.5)
    point = torch.tensor([list(cycle_point)], dtype=torch.float32)
    U_val = compute_quasipotential(model, point)[0]
    U_true_val = system.true_quasipotential(np.array(cycle_point))
    print(f"   On cycle:          U_learned={U_val:.4f}, U_true={U_true_val:.4f} (should be ~0)")

    return rmse, mae


def parse_args():
    parser = argparse.ArgumentParser(description="Quasipotential Model Testing")
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Directory to save output plots (default: %(default)s)",
    )
    parser.add_argument(
        "--ex1_checkpoint",
        type=str,
        default="experiments/example1/checkpoints/checkpoint_10.pth",
        help="Path to Example 1 checkpoint",
    )
    parser.add_argument(
        "--ex2_checkpoint",
        type=str,
        default="experiments/example2/checkpoints/checkpoint_10.pth",
        help="Path to Example 2 checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = _ensure_dir(args.outdir)

    print("Quasipotential Model Testing")
    print("=" * 60)
    print(f"Saving plots to: {outdir}")

    # Test Example 1
    if os.path.exists(args.ex1_checkpoint):
        config1 = quasipotential_ex1.get_config()
        test_example1(args.ex1_checkpoint, config1, outdir)
    else:
        print(f"\nWarning: Checkpoint not found: {args.ex1_checkpoint}")
        print("Skipping Example 1 testing.")

    # Test Example 2
    if os.path.exists(args.ex2_checkpoint):
        config2 = quasipotential_ex2.get_config()
        test_example2(args.ex2_checkpoint, config2, outdir)
    else:
        print(f"\nWarning: Checkpoint not found: {args.ex2_checkpoint}")
        print("Skipping Example 2 testing.")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
