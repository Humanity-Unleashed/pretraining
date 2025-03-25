from optbayesexpt import OptBayesExptNoiseParameter
import wandb
import omegaconf
import hydra
from torch.optim.lr_scheduler import LRScheduler
import numpy as np

import random

use_mock_mode = True

def compute_budget(n_gpus: int, gpu_flops, seconds):
    """
    Returns the budget for a given number of GPUs, flops per GPU, and seconds in floating point operations.
    """

    return n_gpus * gpu_flops * seconds


def flops(model_size, data_size):
    return 6 * model_size * data_size


def cost_function(model_size, data_size, budget):
    # linear
    return flops(model_size, data_size) / budget


class DataAgnosticScheduler(LRScheduler):
    """
    A learning rate schedule with a fixed warmup, peak, and decay period.

    The learning rate starts at 0 and linearly increases to `max_lr` over `warmup_steps` steps.
    It then remains at `max_lr` for `peak_steps` steps before linearly decaying to 0 over `decay_steps` steps.

    See
    https://arxiv.org/abs/2404.06395
    https://arxiv.org/abs/2405.18392
    for more details.
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_steps,
        peak_steps,
        decay_steps,
        last_epoch=10,
        verbose=False,
    ):
        super().__init__(optimizer, last_epoch, verbose)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.decay_steps = decay_steps
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        elif self.current_step < self.peak_steps:
            return self.max_lr
        elif self.current_step < self.decay_steps:
            return (
                self.max_lr
                * (self.decay_steps - self.current_step)
                / (self.decay_steps - self.peak_steps)
            )
        else:
            return 0.0


def init_model(experiment_settings):
    pass


def init_data(experiment_settings):
    pass


def scaling_law_function(settings, params, constants):
    model_size, data_size = settings
    return (
        params["optimal_loss"]
        + params["A"] / model_size ** params["alpha"]
        + params["B"] / (data_size ** params["beta"])
    )

# run an experiment and get the loss
def run_experiment(experiment_settings):
    # TODO: need a way to track number of epochs already ran when loading a checkpoint
    model = init_model(experiment_settings)
    data = init_data(experiment_settings)
    loss = None
    # this is pseudocode
    # loss = optbayesexpt.run_experiment(model, data, experiment_settings)
    if use_mock_mode:
        loss = random.random()
    else:
        raise ValueError(
            "not implemented"
        )  # replace this with calling script to run training
        # run_exp(exp_settings)
        # loss = read_train_log()
    return loss


class ScalingLawsBayesianOptimization(OptBayesExptNoiseParameter):
    def __init__(self, budget, possible_experiment_settings, param_space_size=50000):
        self.param_space_size = param_space_size
        self.budget = budget
        self.params = self.initial_scaling_law_params(param_space_size)

        data_size = np.linspace(
            possible_experiment_settings["data_size"]["min_value"],
            possible_experiment_settings["data_size"]["max_value"],
        )
        model_size = np.linspace(
            possible_experiment_settings["model_size"]["min_value"],
            possible_experiment_settings["model_size"]["max_value"],
        )

        settings = (data_size, model_size)
        constants = ()
        OptBayesExptNoiseParameter.__init__(self, scaling_law_function, settings, self.params, constants)

    def initial_scaling_law_params(self):
        # initial ranges based on Chinchilla scaling laws
        A = np.random.uniform(300, 500, self.param_space_size)
        alpha = np.random.uniform(0, 1, self.param_space_size)
        B = np.random.uniform(300, 500, self.param_space_size)
        beta = np.random.uniform(0, 1, self.param_space_size)
        optimal_loss = np.random.uniform(0, 10, self.param_space_size)

        # estimates of experimental noise
        sigs = np.random.exponential(0.1, self.param_space_size)

        return A, alpha, B, beta, optimal_loss, sigs

    def scaling_law_function(self, settings, params, constants):
        # TODO: make settings the layer numbers, hidden dimensions, etc. instead of parameter count
        # TODO: or calculate the layer numbers, hidden dimensions, etc. from the parameter count
        model_size, data_size = settings
        A, alpha, B, beta, optimal_loss, sigs = params
        return optimal_loss + A / (model_size**alpha) + B / (data_size**beta)
    
    def cost_estimate(self):
        """Returns cost of experiments, used in utility function"""
        return cost_function(self.allsettings[0], self.allsettings[1], self.budget)

    def iterate(self):
        # choose experiment settings with optbayes
        # run an experiment
        # refine the scaling law with the loss and optbayes
        # repeat until you use up the budget
        
        chosen_settings = self.opt_setting()

        loss = run_experiment(chosen_settings)

        compute += flops(chosen_settings[0], chosen_settings[1])

        measure = (chosen_settings, loss)
        self.pdf_update(measure)
        
    def run(self):
        compute = 0
        while compute < self.budget:
            self.iterate()


def main():
    exp_settings_dict = {
        # Avoid making the minimum value too small, because scaling laws break down at small values.
        # This needs more work
        "data_size": {"min_value": 10**3, "max_value": 10**6},
        "model_size": {"min_value": 10**6, "max_value": 10**9},
    }
    budget = compute_budget(1, 10**9, 60 * 60 * 24 * 7)
    exp = ScalingLawsBayesianOptimization(budget, param_space_size=50000)
    exp.iterate(exp_settings_dict)


if __name__ == "__main__":
    main()
