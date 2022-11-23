def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    config['mode'] = "train"
    config['use_wandb'] = False
    config['use_cuda'] = True
    config['log_dir'] = "/AS_clean/tuft_fs/logs"
    config['model_load_dir'] = None # required for test-only mode

    # Hyperparameters for dataset. 
    config['view'] = 'all' # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'mod_severe'
    config['view_scheme_name'] = 'three_class'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    config['num_classes_diagnosis'] = 3
    config['num_classes_view'] = 3

    # Hyperparameters for models.
    config['model'] = "FTC_res18" # wideresnet
    config['pretrained'] = False
    config['restore'] = False
    config['loss_type'] = 'cross_entropy' # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR

    # Hyperparameters for training.
    config['batch_size'] = 4
    config['num_epochs'] = 500
    config['lr'] = 0.00007  #1e-4 for Resnet2+1D, 1e-5 for FTC
    config['sampler'] = 'random' # imbalanaced sampling based on AS/bicuspid/random
 
    return config