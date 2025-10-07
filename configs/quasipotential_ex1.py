"""onfiguration file for the first quasipotential experiment - configs/quasipotential_ex1.py"""

import ml_collections
import torch


def get_config():
   config = ml_collections.ConfigDict()
   
   # Training
   config.training = training = ml_collections.ConfigDict()
   training.batch_size = 128 
   training.n_iters = 50000 
   training.snapshot_freq = 5000 
   training.log_freq = 100
   training.eval_freq = 1000
   training.reduce_mean = True
   training.likelihood_weighting = False
   training.continuous = True
   training.sde = 'quasipotential' 
   training.snapshot_freq_for_preemption = 5000
   training.snapshot_sampling = False
   
   # Data configuration
   config.data = data = ml_collections.ConfigDict()
   data.dataset = 'quasipotential'  # This will trigger the right choise for the dataset
   data.example = 'example1'
   data.dim = 3 # Dimention of the system
   data.n_trajectories = 2000
   data.T = 5.0 # Time horizont
   data.dt = 0.01 # Time step
   data.domain = [[-2, 2], [-2, 2], [-2, 2]]
   data.num_channels = 3  # Same as the dim, for compatibility
   data.image_size = 1 # dummy value
   data.uniform_dequantization = False
   data.centered = False  # Keep data in [0,1] range
   data.random_flip = False  # No random flips for 2D point data
   
   # Model configuration
   config.model = model = ml_collections.ConfigDict()
   model.name = 'quasipotential_mlp'
   model.dim = 3 # Input/output dimentions
   model.hidden_dims = [128, 128]
   model.dropout = 0.0 # form 0.05 to no dropout for full capacity
   model.activation = "tanh"
   model.lambda_orth = 0.5  # Orthogonality loss weight
   model.embedding_type = 'none'
   model.ema_rate = 0.9999
   
   # Optimization
   config.optim = optim = ml_collections.ConfigDict()
   optim.weight_decay = 0
   optim.optimizer = 'Adam'
   optim.lr = 1e-3
   optim.beta1 = 0.9
   optim.eps = 1e-8
   optim.warmup = 0
   optim.grad_clip = 1.0 
   
   # Evaluation
   config.eval = evaluate = ml_collections.ConfigDict()
   evaluate.begin_ckpt = 1
   evaluate.end_ckpt = 10
   evaluate.batch_size = 1000
   evaluate.enable_sampling = False
   evaluate.enable_loss = True

   # Device
   config.seed = 42
   # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   config.device = 'cpu'

   return config