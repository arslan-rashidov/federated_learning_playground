import os

from inspect import signature, Parameter

import pandas as pd
import torch
from torch import nn
import torch.utils.data

### TODO: Change names of local required parameters
get_dataset_fn_not_required_params = {
    'with_split': str
}

train_fn_not_required_params = {
    'model': nn.Module,
    'train_set': torch.utils.data.Dataset,
    'valid_set': torch.utils.data.Dataset,
}

test_fn_not_required_params = {
    'model': nn.Module,
    'test_set': torch.utils.data.Dataset,
    'return_output': bool
}


class Client:
    def __init__(self,
                 load_model_fn,
                 model_global_parameters=None,
                 dataset_global_parameters=None,
                 train_global_parameters=None,
                 test_global_parameters=None,
                 train_fn=None,
                 test_fn=None,
                 get_dataset_fn=None,
                 initial_weights_path=None):

        self.model_global_parameters = model_global_parameters
        self.dataset_global_parameters = dataset_global_parameters
        self.train_global_parameters = train_global_parameters
        self.test_global_parameters = test_global_parameters

        self.load_model_fn = load_model_fn
        self.get_dataset_fn = get_dataset_fn
        self.train_fn = train_fn
        self.test_fn = test_fn

        if self.get_dataset_fn:
            self.get_dataset_user_required_parameters = get_fn_parameters(get_dataset_fn,
                                                                          [*list(
                                                                              get_dataset_fn_not_required_params.keys()),
                                                                           *list(
                                                                               self.dataset_global_parameters.keys())])
        if self.train_fn:
            self.train_user_required_parameters = get_fn_parameters(train_fn,
                                                                    [*list(train_fn_not_required_params.keys()),
                                                                     *list(self.train_global_parameters.keys())])

        if self.test_fn:
            self.test_user_required_parameters = get_fn_parameters(test_fn,
                                                                   [*list(test_fn_not_required_params.keys()),
                                                                    *list(self.test_global_parameters.keys())])

        self.model = self.load_model_fn(
            **self.model_global_parameters) if self.model_global_parameters else self.load_model_fn()

        if initial_weights_path is not None:
            self.set_weights(weights_path=initial_weights_path)

        self.train_set, self.valid_set, self.test_set = None, None, None
        self.device = None

    def set_weights(self, weights=None, weights_path=None):
        assert weights is not None or weights_path is not None, "Either weights or weights_path must be provided"
        assert not (
                    weights is not None and weights_path is not None), "Only one of weights or weights_path can be provided"

        if weights_path:
            weights = torch.load(weights_path)

        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def fit(self, **kwargs):
        get_dataset_user_parameters = {}
        train_user_parameters = {}
        test_user_parameters = {}

        for arg, val in kwargs.items():
            if arg in [param['name'] for param in self.get_dataset_user_required_parameters]:
                get_dataset_user_parameters[arg] = val
            elif arg in [param['name'] for param in self.train_user_required_parameters]:
                train_user_parameters[arg] = val
            elif arg in [param['name'] for param in self.test_user_required_parameters]:
                test_user_parameters[arg] = val

        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.train_set is None:
            sets = self.get_dataset_fn(with_split=True, **self.dataset_global_parameters, **get_dataset_user_parameters)
            if len(sets) == 3:
                self.train_set, self.valid_set, self.test_set = sets
            elif len(sets) == 2:
                self.train_set, self.test_set = sets
            elif len(sets) == 1:
                self.test_set = sets

        fit_metrics = []
        evaluate_metrics = []

        if self.train_set and self.valid_set:
            fit_metrics, self.model = self.train_fn(model=self.model, train_set=self.train_set,
                                                    valid_set=self.valid_set,
                                                    **self.train_global_parameters, **train_user_parameters)
        elif self.train_set:
            fit_metrics, self.model = self.train_fn(model=self.model, train_set=self.train_set,
                                                    **self.train_global_parameters, **train_user_parameters)

        if self.test_set:
            evaluate_metrics = self.test_fn(model=self.model, test_set=self.test_set, return_output=False,
                                            **self.test_global_parameters, **test_user_parameters)

        trained_weights = self.get_weights()
        metrics = [*fit_metrics, *evaluate_metrics]

        fit_additional_data = {
            'train_num_examples': [len(self.train_set)]
        }

        save_output(trained_weights, metrics, additional_data=fit_additional_data)

    def evaluate(self, **kwargs):
        get_dataset_user_parameters = {}
        test_user_parameters = {}

        for arg, val in kwargs.items():
            if arg in [param['name'] for param in self.get_dataset_user_required_parameters]:
                get_dataset_user_parameters[arg] = val
            elif arg in [param['name'] for param in self.test_user_required_parameters]:
                test_user_parameters[arg] = val

        eval_set = self.test_set

        if 'dataset_path' in get_dataset_user_parameters:
            eval_set = self.get_dataset_fn(with_split=False, **self.dataset_global_parameters,
                                           **get_dataset_user_parameters)

        assert eval_set is not None

        eval_metrics, eval_output = None, None

        if 'return_output' in kwargs:
            eval_metrics, eval_output = self.test_fn(model=self.model, test_set=eval_set, return_output=True,
                                                     **self.test_global_parameters)
        else:
            eval_metrics = self.test_fn(model=self.model, test_set=eval_set, return_output=False,
                                        **self.test_global_parameters)

        if eval_output:
            save_output(metrics=eval_metrics, eval_output=eval_output)
        else:
            save_output(metrics=eval_metrics)


def get_fn_parameters(fn, excluded_parameters: list):
    sig = signature(fn)

    parameters = []

    for param_name in sig.parameters:
        if param_name not in excluded_parameters:
            parameters.append({
                'name': param_name,
                'type': None if sig.parameters[param_name].annotation is Parameter.empty else sig.parameters[
                    param_name].annotation,
                'default': None if sig.parameters[param_name].default is Parameter.empty else sig.parameters[
                    param_name].default
            })

    return parameters


def save_weights(weights, path):
    torch.save(weights, path)


def save_metric(metric, path):
    metric_df = metric.get_dataframe()
    metric_df.to_csv(path, index=False)


def save_output(weights=None, metrics=None, eval_output=None,
                additional_data=None):  # TODO: Decomposite for fit and eval output
    output_parent_directory_path = "output/"

    def get_experiment_output_directory(output_parent_directory_path):
        current_experiment = 0
        if os.path.isdir(output_parent_directory_path):
            experiments_count = len(os.listdir(output_parent_directory_path))
            current_experiment = experiments_count + 1
        else:
            os.mkdir(output_parent_directory_path)
            current_experiment = 1
        current_experiment_output_directory_path = f"{output_parent_directory_path}experiment_{current_experiment}/"
        os.mkdir(current_experiment_output_directory_path)
        return current_experiment_output_directory_path

    output_directory_path = get_experiment_output_directory(output_parent_directory_path)

    if weights:
        weights_output_directory_path = output_directory_path + "weights/"
        os.mkdir(weights_output_directory_path)

        weights_path = weights_output_directory_path + "weights.pth"

        save_weights(weights, weights_path)

    if metrics:
        metrics_output_directory_path = output_directory_path + "metrics/"
        os.mkdir(metrics_output_directory_path)
        for metric in metrics:
            metric_path = metrics_output_directory_path + metric.name + '.csv'
            save_metric(metric, metric_path)

    if eval_output:
        eval_output_directory_path = output_directory_path + "eval_output/"
        os.mkdir(eval_output_directory_path)

        eval_output_path = eval_output_directory_path + "output.csv"
        eval_output_df = pd.DataFrame(data={'value': eval_output})
        eval_output_df.to_csv(eval_output_path, index=False)

    if additional_data:
        additional_data_directory_path = output_directory_path + "additional_data/"
        os.mkdir(additional_data_directory_path)

        additional_data_path = additional_data_directory_path + "additional_data.csv"
        additional_data_df = pd.DataFrame(data=additional_data)
        additional_data_df.to_csv(additional_data_path, index=False)
