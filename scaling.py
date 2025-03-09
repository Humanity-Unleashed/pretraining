import optbayesexpt
import wandb
import omegaconf
import hydra
from torch.optim.lr_scheduler import LRScheduler

# you can change the parameters if it makes sense
def compute_budget(n_gpus, time):
    pass

# based on compute
def cost_function(model_size, data_size, budget):
    pass

class DataAgnosticScheduler(LRScheduler):
    def __init__(self, optimizer, last_epoch = ..., verbose = ...):
        super().__init__(optimizer, last_epoch, verbose)
    
def init_model(experiment_settings):
    pass

def init_data(experiment_settings):
    pass

# run an experiment and get the loss
def run_experiment(experiment_settings):
    model = init_model(experiment_settings)
    data = init_data(experiment_settings)
    loss = None
    # this is pseudocode
    # loss = optbayesexpt.run_experiment(model, data, experiment_settings)
    return loss

# for our functional hypothesis
class ScalingLaw:
    def __init__(self):
        pass
    
    def __call__(self, model_size, data_size):
        pass

def iterate(possible_experiment_settings):
    # choose experiment settings with optbayes
    # run an experiment
    # refine the scaling law with the loss and optbayes
    # repeat until you use up the budget
    pass
