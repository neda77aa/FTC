def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    config['mode'] = "test"
    config['use_wandb'] = False
    config['use_cuda'] = True
    config['log_dir'] = "/AS_Neda/FTC/logs"
    config['model_load_dir'] = None # required for test-only mode

    # Hyperparameters for dataset. 
    config['view'] = 'all' # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    config['num_classes'] = 4
    
    # Hyperparameters for bicuspid valve branch
    # config['bicuspid_weight'] = 1.0 # default is 1.0
    
    # Hyperparameters for Contrastive Learning
    config['cotrastive_method'] = 'CE' #'CE'/'SupCon'/'SimCLR'/Linear'
    config['feature_dim'] = 1024
    config['temp'] = 0.1

    # Hyperparameters for models.
    config['model'] = "FTC_TAD" # r2plus1d_18/x3d/resnet50/slowfast/tvn/FTC
    config['pretrained'] = False
    config['restore'] = True
    config['loss_type'] = 'cross_entropy' # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR

    # Hyperparameters for training.
    config['batch_size'] = 16
    config['num_epochs'] = 110
    config['lr'] = 1e-4  #1e-4 for Resnet2+1D, 1e-5 for FTC
    config['sampler'] = 'random' # imbalanaced sampling based on AS/bicuspid/random
 
    return config