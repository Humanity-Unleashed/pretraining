import optbayesexpt
import wandb
import omegaconf
import hydra
from torch.optim.lr_scheduler import LRScheduler

import random

use_mock_mode = True

# you can change the parameters if it makes sense
def compute_budget(n_gpus, time):
    return n_gpus*time

# based on compute
def cost_function(model_size, data_size, budget):
    return model_size*data_size*budget

class DataAgnosticScheduler(LRScheduler):
    def __init__(self, optimizer, last_epoch=10 verbose =False):
        super().__init__(optimizer, last_epoch, verbose)
    
def init_model(experiment_settings):
    pass

def init_data(experiment_settings):
    pass

def scaling_law_function(settings, params, constants):
    model_size,data_size = settings
    return params["optimal_loss"] + \
        params["A"] / model_size ** params["alpha"] + \
        params["B"]/(data_size ** params["beta"])

# for our functional hypothesis
class ScalingLaw:
    def __init__(self):
        self.params = {
            "A": None,
            "alpha": None,
            "B": None,
            "beta": None,
            "optimal_loss": None,
        }
        for p in self.params:
            self.params[p] = np.linspace(0,1000000,100)
    
    def __call__(self, model_size, data_size):
        return scaling_law_function((model_size, data_size), self.params, ())

my_scale_law = ScalingLaw()

# run an experiment and get the loss
def run_experiment(experiment_settings):
    model = init_model(experiment_settings)
    data = init_data(experiment_settings)
    loss = None
    # this is pseudocode
    # loss = optbayesexpt.run_experiment(model, data, experiment_settings)
    if use_mock_mode:
        loss = random.random()
    else:
        raise ValueError("not implemented") #replace this with calling script to run training
        #run_exp(exp_settings)
        #loss = read_train_log()
    return loss

def iterate(possible_experiment_settings):
    # choose experiment settings with optbayes
    # run an experiment
    # refine the scaling law with the loss and optbayes
    # repeat until you use up the budget
    data_size = np.linspace(possible_experiment_settings["data_size"]["min_value"],possible_experiment_settings["data_size"]["max_value"])
    model_size = np.linspace(possible_experiment_settings["model_size"]["min_value"],possible_experiment_settings["model_size"]["max_value"])

    settings = (data_size,model_size) 
    parameters = my_scale_law.params
    constants = ()

    my_obe = OptBayesExpt(scaling_law_function, settings, parameters, constants, scale=False)

    num_exp = possible_experiment_settings["NUM_EXP"]
    use_opt_settings = possible_experiment_settings["USE_OPT"]
    for i in range(NUM_EXPS):
        if optimal:
            chosen_settings = my_obe.opt_setting()
        else:
            chosen_settings = my_obe.good_setting(pickiness=0.8)

        loss = run_experiment(chosen_settings)
        measure = (chosen_settings,loss)
        my_obe.update_pdf(measure)
        
def main():
    exp_settings_dict = {
        "data_size": {"min_value",1,"max_value":1000},
        "model_size": {"min_value":10**6,"max_value":10**9},
        "NUM_EXP": 10,
        "USE_OPT": True
    }
    iterate(exp_settings_dict)

if __name__=="__main__":
    main()