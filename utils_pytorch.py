from copy import deepcopy
import os
from collections import namedtuple

import numpy as np
import torch
from torch.jit import trace


ExperimentConfig = namedtuple('ExperimentConfig',
                              ('base_path', 'experiment_name', 'dataset_name'))


def get_config():
    return ExperimentConfig(base_path=os.getenv("AICROWD_OUTPUT_PATH","../scratch/shared"),
                            experiment_name=os.getenv("AICROWD_EVALUATION_NAME", "experiment_name"),
                            dataset_name=os.getenv("AICROWD_DATASET_NAME", "cars3d"))


def get_model_path(base_path=None, experiment_name=None, make=True):
    base_path = os.getenv("AICROWD_OUTPUT_PATH","../scratch/shared") \
        if base_path is None else base_path
    experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name") \
        if experiment_name is None else experiment_name
    model_path = os.path.join(base_path, experiment_name, 'representation', 'pytorch_model.pt')
    if make:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return model_path


def export_model(model, path=None, input_shape=None):
    path = get_model_path() if path is None else path
    model = deepcopy(model).cpu().eval()
    if not isinstance(model, torch.jit.ScriptModule):
        assert input_shape is not None, "`input_shape` must be provided since model is not a " \
                                        "`ScriptModule`."
        traced_model = trace(model, torch.zeros(*input_shape))
    else:
        traced_model = model
    torch.jit.save(traced_model, path)
    return path


def import_model(path=None):
    path = get_model_path() if path is None else path
    return torch.jit.load(path)


def make_representor(model, cuda=False):
    model = deepcopy(model)
    model = model.cuda() if cuda else model.cpu()

    # Define the representation function
    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray."
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor."
        # Convert from NHWC to NCHW
        x = np.moveaxis(x, 3, 1)
        # Convert to torch tensor and evaluate
        x = torch.from_numpy(x).to('cuda' if cuda else 'cpu')
        with torch.no_grad():
            y = model(x)
        y = y.cpu().numpy()
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent


if __name__ == '__main__':
    import torch.nn as nn
    model = nn.Linear(10, 10)
    path = export_model(model,
                        path='/Users/nrahaman/Documents/Python/'
                             'neurips2019_disentanglement_challenge_starter_kit/'
                             'scratch/test.pt',
                        input_shape=(1, 10))
    model2 = import_model(path)
    pass