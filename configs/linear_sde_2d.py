"""Configuration for Linear SDE with Residual MLP - configs/linear_sde_2d.py"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 256
    training.n_iters = 300000
    training.snapshot_freq = 25000
    training.log_freq = 500
    training.eval_freq = 2500
    training.reduce_mean = True
    training.likelihood_weighting = False
    training.continuous = True
    training.sde = 'linear'
    training.snapshot_freq_for_preemption = 10000
    training.snapshot_sampling = True
    
    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'toy2d'
    data.toy_type = 'gaussian'
    data.image_size = 1
    data.num_channels = 2
    data.uniform_dequantization = False
    data.centered = False
    data.random_flip = False
    
    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'simple_mlp'
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000 
    model.beta_min = 0.1
    model.beta_max = 20
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128  
    model.num_res_blocks = 4  # Can go deeper with residuals
    model.ch_mult = (1, 2, 2, 2)
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
    model.fourier_scale = 4
    model.conv_size = 3
    
    # LinearSDE parameters
    model.A_matrix = [[2.0, 1.0], [0.0, 1.0]]
    model.epsilon = 0.5
    
    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4  # Conservative starting LR
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0
    optim.lr_decay = 'none'  # No decay after warmup ('none', 'cosine', 'exponential')
    
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
    evaluate.end_ckpt = 12
    evaluate.batch_size = 1000
    evaluate.enable_sampling = True
    evaluate.num_samples = 5000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'
    
    config.seed = 42
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.device = 'cpu'
    
    return config