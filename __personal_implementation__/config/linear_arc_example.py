''' Arc example with LINEAR drift matrix A = [[-a, b], [-b, -a]] '''
import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.n_epochs = 3000
    training.lr = 1e-3
    training.weight_decay = 0.0
    training.sigma_dn = 0.1

    # Data / SDE Parameters 
    config.data = data = ml_collections.ConfigDict()
    data.dim = 2                # System dimensions (must be 2D for this drift)
    data.n_points = 10000          
    data.dt = 0.01              
    data.T = 20.0                
    data.epsilon = 0.15         # Noise
    
    # Manifold 
    config.manifold = manifold = ml_collections.ConfigDict()
    manifold.type = 'arc'       
    manifold.radius = 3.0       
    manifold.arc_fraction = 0.35 
    
    # Drift: Linear matrix A = [[-a, b], [-b, -a]]
    config.drift = drift = ml_collections.ConfigDict()
    drift.type = 'linear'       # NEW: linear drift
    drift.a = 1.0               # Contraction coefficient (a > 0 → moves toward origin)
    drift.b = 2.0               # Rotation coefficient (b ≠ 0 → rotation)
    
    # Note: This creates a spiral toward the origin!
    # - If b = 0: pure contraction (like attractive drift)
    # - If a = 0: pure rotation
    # - Both nonzero: spiral inwaard
    
    # Visualization
    config.visualization = vis = ml_collections.ConfigDict()
    vis.save_every = 1         
    
    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ScoreNet'  
    model.hidden_dim = 128      
    model.n_layers = 3          
    model.time_embedding_dim = 32  
    model.save_path = 'models/linear_arc_trained.pt'
    
    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 1e-3
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    
    # Sampling (for reverse SDE)
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'euler_maruyama'
    sampling.n_steps = 500
    
    # Evaluation
    config.eval = evaluation = ml_collections.ConfigDict()
    evaluation.batch_size = 100
    evaluation.metrics = ['distance', 'mse']
    
    # Device
    config.seed = 42
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config.name = 'Arc with Linear Drift (Spiral)'
    
    return config