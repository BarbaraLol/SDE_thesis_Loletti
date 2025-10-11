''' Points in a arc segment collapsing towards the center '''
import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.n_epochs = 1000
    training.lr = 1e-3          # Learning rate
    training.weight_decay = 0.0

    # Data / SDE Parameters 
    config.data = data = ml_collections.ConfigDict()
    data.dim = 2                # System dimentions
    data.n_points = 60          
    data.dt = 0.01              
    data.T = 5.0                
    data.epsilon = 0.15         # Noise
    
    # Manifold 
    config.manifold = manifold = ml_collections.ConfigDict()
    manifold.type = 'arc'       
    manifold.radius = 3.0       
    manifold.arc_fraction = 0.35 
    
    # Drift 
    config.drift = drift = ml_collections.ConfigDict()
    drift.type = 'attractive'   
    drift.strength = 0.8       # Attraction intensity towards the center
    # other parameters (just if the problem requires them)
    drift.omega = 1.0           # For rotational drift
    drift.attractive_strength = 0.5  # For combined drift
    drift.rotational_omega = 0.5     # For combined drift
    drift.a = 1.0               # For double_well drift
    drift.b = 1.0               # For double_well drift
    drift.lambda_stable = 1.0   # For saddle drift
    drift.lambda_unstable = 1.0 # For saddle drift
    drift.theta = 1.0           # For ornstein_uhlenbeck drift
    
    # Visualization
    config.visualization = vis = ml_collections.ConfigDict()
    vis.save_every = 1         
    
    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ScoreNet'  
    model.hidden_dim = 128      
    model.n_layers = 3          
    model.time_embedding_dim = 32  
    
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
    
    
    config.name = 'Arc to Center'
    
    return config