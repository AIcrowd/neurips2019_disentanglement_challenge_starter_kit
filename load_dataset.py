import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import os
os.environ.update({'DISENTANGLEMENT_LIB_DATA': os.path.join(os.path.dirname(__file__),
                                                            'scratch',
                                                            'dataset')})

# noinspection PyUnresolvedReferences
from disentanglement_lib.data.ground_truth import cars3d, dsprites, mpi3d, norb, shapes3d


class DLIBDataset(Dataset):
    """
    No-bullshit data-loading from Disentanglement Library, but with a few sharp edges.
    """
    # Registry of available datasets and the respective modules they reside in.

    DATASETS = {
        'Cars3D': 'cars3d',
        'DSprites': 'dsprites',
        'ColorDSprites': 'dsprites',
        'NoisyDSprites': 'dsprites',
        'ScreamDSprites': 'dsprites',
        'MPI3D': 'mpi3d',
        'SmallNORB': 'norb',
        'Shapes3D': 'shapes3d'
    }

    def __init__(self, name, seed=0, iterator_len=50000):
        assert name in self.DATASETS, f"name must be one of: {list(self.DATASETS.keys())}"
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.iterator_len = iterator_len
        self.dataset = self.load_dataset()

    def load_dataset(self):
        dataset_module = globals().get(self.DATASETS[self.name])
        assert dataset_module is not None
        return getattr(dataset_module, self.name)()

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        assert item < self.iterator_len
        output = self.dataset.sample_observations(1, random_state=self.random_state)[0]
        # Convert output to CHW from HWC
        return torch.from_numpy(np.moveaxis(output, 2, 0))


if __name__ == '__main__':
    dlib_dataset = DLIBDataset('Cars3D')
    print(dlib_dataset[0].shape)
    # ^ prints `torch.Size([3, 64, 64])`

