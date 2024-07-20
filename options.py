from utility_functions import * 

# Configuration for the model training and evaluation.
# Initialize a dictionary to store all experiment parameters.
args = {
    'exp_name': 'exp_1',  # Name of the experiment.
    'batch_size': 100,  # Number of samples in each batch.
    'epochs': 100,  # Total number of training epochs.
    'past_length': 8,  # Number of past frames to consider for the prediction.
    'future_length': 12,  # Number of future frames to predict.
    'no_cuda': False,  # Flag to disable CUDA even if available.
    'seed': -1,  # Seed for random number generation. -1 means no specific seed.
    'log_interval': 1,  # Frequency of logging training status.
    'test_interval': 1,  # Frequency of testing the model.
    'outf': 'n_body_system/logs',  # Output directory for logs.
    'lr': 1e-6,  # Learning rate for the optimizer.
    'epoch_decay': 2,  # Number of epochs after which learning rate will decay.
    'lr_gamma': 0.8,  # Learning rate decay factor.
    'nf': 64,  # Number of features.
    'model': 'egnn_vel',  # Model type to be used.
    'attention': 0,  # Whether to use attention mechanism.
    'n_layers': 4,  # Number of layers in the neural network.
    'degree': 2,  # Degree parameter for some models.
    'channels': 64,  # Number of channels in models.
    'max_training_samples': 3000,  # Maximum number of training samples to consider.
    'dataset': 'nbody',  # Dataset to use.
    'sweep_training': 0,  # Whether to use parameter sweeping in training.
    'time_exp': 0,  # Flag for time experiment.
    'weight_decay': 1e-12,  # Weight decay to prevent overfitting.
    'div': 1,  # Division factor for something (ambiguous without context).
    'norm_diff': False,  # Whether to normalize differences.
    'tanh': False,  # Whether to apply tanh activation.
    'subset': 'eth',  # Subset of data to be used.
    'model_save_dir': '/kaggle/working/saved_models',  # Directory to save trained models.
    'scale': 1,  # Scale factor for inputs/outputs.
    'apply_decay': False,  # Whether to apply decay to learning rate.
    'res_pred': False,  # Whether to use residual predictions.
    'supervise_all': False,  # Whether all layers are supervised.
    'model_name': 'eth_ckpt_best',  # Name to save the model checkpoint.
    'test_scale': 1,  # Scaling for testing phase.
    'test': False,  # Whether to run tests.
    'vis': False  # Whether to enable visualization.
}

# Create a class to mimic argparse's namespace functionality.
class ArgsNamespace:
    def __init__(self, adict):
        self.__dict__.update(adict)  # Update the object's dictionary with the passed dictionary.

# Instantiate the ArgsNamespace class with the args dictionary.
args = ArgsNamespace(args)
args.cuda = torch.cuda.is_available()  # Check if CUDA is available and update the args object.