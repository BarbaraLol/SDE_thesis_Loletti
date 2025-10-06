"""Fixed configuration file for Linear SDE 2D training - configs/linear_sde_2d.py"""

import ml_collections
import torch


def get_config():
   config = ml_collections.ConfigDict()
   
   # Training
   config.training = training = ml_collections.ConfigDict()
   training.batch_size = 128 # 256
   training.n_iters = 500000 # 250000
   training.snapshot_freq = 25000 # 10000
   training.log_freq = 500 # 100
   training.eval_freq = 2500 # 1000
   training.reduce_mean = True
   training.likelihood_weighting = True
   training.continuous = True
   training.sde = 'linear'  # Use LinearSDE
   training.snapshot_freq_for_preemption = 10000 # 5000
   training.snapshot_sampling = True
   
   # Data configuration
   config.data = data = ml_collections.ConfigDict()
   config.data.image_size = 2  # Not spatial dimensions, just feature dimensions
   data.dataset = 'toy2d'  # This will trigger get_toy_2d_dataset
   data.toy_type = 'gaussian'  # Options: 'gaussian', 'circle', 'spiral'
   data.image_size = 1  # Minimal spatial dimensions for (N, 2, 1, 1) format
   data.num_channels = 2  # 2 channels for x,y coordinates
   data.uniform_dequantization = False
   data.centered = False  # Keep data in [0,1] range
   data.random_flip = False  # No random flips for 2D point data
   
   # Model configuration - using SimpleMLP
   config.model = model = ml_collections.ConfigDict()
   model.name = 'simple_mlp'  # Use SimpleMLP instead of CNN
   model.sigma_min = 0.01
   model.sigma_max = 50
   model.num_scales = 1000 
   model.beta_min = 0.1
   model.beta_max = 20
   model.dropout = 0.0 # form 0.05 to no dropout for full capacity
   model.embedding_type = 'fourier'
   model.scale_by_sigma = True
   model.ema_rate = 0.999
   model.normalization = 'GroupNorm'
   model.nonlinearity = 'swish'
   model.nf = 256  # Number of features for MLP
   model.ch_mult = (1, 2, 2, 2)  # Not used by MLP but kept for compatibility
   model.num_res_blocks = 8
   model.attn_resolutions = (16,)
   model.resamp_with_conv = True
   model.conditional = True
   model.fir = True
   model.fir_kernel = [1, 3, 3, 1]
   model.skip_rescale = True
   model.resblock_type = 'biggan'
   model.progressive = 'none'
   model.progressive_input = 'residual'
   model.progressive_combine = 'sum'
   model.attention_type = 'ddpm'
   model.init_scale = 0.0
   model.fourier_scale = 16
   model.conv_size = 3
   
   # LinearSDE specific parameters with proper non-equilibrium matrix
   model.A_matrix = [[1.0, 0.8], [0.3, 1.2]]  # Non-symmetric matrix
   model.epsilon = 0.5  # Noise strength
   
   # Optimization
   config.optim = optim = ml_collections.ConfigDict()
   optim.weight_decay = 1e-5
   optim.optimizer = 'Adam'
   optim.lr = 5e-4 
   optim.beta1 = 0.9
   optim.eps = 1e-8
   optim.warmup = 10000
   optim.grad_clip = 2.0 
   
   # Sampling
   config.sampling = sampling = ml_collections.ConfigDict()
   sampling.method = 'pc'
   sampling.predictor = 'euler_maruyama'
   sampling.corrector = 'none'
   sampling.n_steps_each = 1
   sampling.noise_removal = True
   sampling.probability_flow = False
   sampling.snr = 0.15
   
   # Evaluation
   config.eval = evaluate = ml_collections.ConfigDict()
   evaluate.begin_ckpt = 1
   evaluate.end_ckpt = 20
   evaluate.batch_size = 1000
   evaluate.enable_sampling = True
   evaluate.num_samples = 5000
   evaluate.enable_loss = True
   evaluate.enable_bpd = False
   evaluate.bpd_dataset = 'test'
   # Device
   # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   config.device = 'cpu'
   return config