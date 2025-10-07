# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
import sde_lib
from sde_lib import VESDE, VPSDE, LinearSDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


# def optimization_manager(config):
#   """Returns an optimize_fn based on `config`."""
  
#   # Learning rate schedule with warmup and cosine decay
#   warmup = config.optim.warmup
#   total_steps = config.training.n_iters
#   base_lr = config.optim.lr
  
#   def optimization_manager(config):
#     """Returns an optimize_fn based on `config`."""

#     def optimize_fn(optimizer, params, step, lr=config.optim.lr,
#                     warmup=config.optim.warmup,
#                     grad_clip=config.optim.grad_clip):
#       """Optimizes with warmup and gradient clipping."""
      
#       # Apply warmup
#       if warmup > 0 and step < warmup:
#         current_lr = lr * (step / warmup)
#         for g in optimizer.param_groups:
#           g['lr'] = current_lr
#       else:
#         # After warmup, use base learning rate (no decay)
#         for g in optimizer.param_groups:
#           g['lr'] = lr
      
#       # Gradient clipping
#       if grad_clip >= 0:
#         torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
      
#       optimizer.step()

#     return optimize_fn

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""
  
  # Get decay type from config (default: none)
  decay_type = getattr(config.optim, 'lr_decay', 'none')

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup, optional decay, and gradient clipping."""
    
    # Calculate learning rate
    if step < warmup and warmup > 0:
      # Linear warmup
      current_lr = lr * (step / warmup)
    else:
      # After warmup - no decay by default
      current_lr = lr
    
    # Set learning rate
    for g in optimizer.param_groups:
      g['lr'] = current_lr
    
    # Gradient clipping
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    
    optimizer.step()

  return optimize_fn

def get_lr_schedule_fn(config):
  """Create learning rate schedule with warmup and cosine decay."""
  warmup = config.optim.warmup
  total_steps = config.training.n_iters
  base_lr = config.optim.lr
  
  def schedule_fn(step):
    if step < warmup:
      # Linear warmup
      return base_lr * (step / warmup)
    else:
      # Cosine decay
      progress = (step - warmup) / (total_steps - warmup)
      return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
  
  return schedule_fn

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    
    # Handle different SDE types for variance scaling
    if isinstance(sde, sde_lib.LinearSDE):
      # For LinearSDE, std has shape (batch_size, 2, 1, 1) - anisotropic noise
      perturbed_data = mean + std * z
    else:
      # For other SDEs (VP, VE), std is 1D and needs broadcasting
      perturbed_data = mean + std[:, None, None, None] * z
        
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      # Standard score matching loss for LinearSDE
      if isinstance(sde, sde_lib.LinearSDE):
        # CORRECTED: For anisotropic noise, the target is -z/std
        # But we need to be careful with the loss formulation
        # Loss: E[||score * std + z||²]
        losses = torch.square(score * std + z)
      else:
        losses = torch.square(score * std[:, None, None, None] + z)
          
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      # Likelihood weighting version
      if isinstance(sde, sde_lib.LinearSDE):
        # For LinearSDE, we need to weight by g²
        # g(t) = epsilon (constant in time for your case)
        g2 = sde.epsilon ** 2
        # Loss: E[||score + z/std||² * g²]
        # Need to handle division carefully for anisotropic std
        losses = torch.square(score + z / (std + 1e-8))
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
      else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / (std[:, None, None, None] + 1e-8))
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_quasipotential_loss_fn(config, train, reduce_mean=True):
  ''' Definition of the loss for the quasipotential learning'''
  def loss_fn(model, batch):
    x = batch['image']  # (batch, dim, 1, 1)
    f_true = batch['vector_field']  # (batch, dim)
    
    # Get model predictions
    f_pred = model(x, None)  # (batch, dim, 1, 1)
    f_pred = f_pred.squeeze(-1).squeeze(-1)
    
    # Dynamics reconstruction loss
    loss_dyn = torch.mean((f_pred - f_true)**2)
    
    # Orthogonality loss
    x_flat = x.squeeze(-1).squeeze(-1)
    grad_v = model.compute_grad_V(x_flat)
    g = model.compute_g(x_flat)
    inner_prod = torch.sum(grad_v * g, dim=1)
    loss_orth = torch.mean(inner_prod**2)
    
    # Total loss
    lambda_orth = config.model.lambda_orth
    loss = loss_dyn + lambda_orth * loss_orth
    
    return loss
  
  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
  
  # if config.training.sde == 'quasipotential':
  #   loss_fn = get_quasipotential_loss_fn(config, train, reduce_mean)

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
