import numpy as np
import torch as t

from .dataset import Dataset, Entity
from .loader_aug import Loader_aug


class Loader_test(object):
    """Memory-light replacement of Loader_aug for the test path (anomaly_types=['normal'], no shuffle).

    Loader_aug materializes Y, Z and anomaly_mask for every window (~3 x n_windows x n_features x window_size
    floats, ~12 GB for the SMAP test set with step 1). This loader slices the same windows on the fly from each
    entity's array, so batches (keys, order, dtype, values) are identical to Loader_aug with 'normal' windows:
    Y = Z = Y[:, start:start+window_size] as float32, anomaly_mask = ones, label = one-hot('normal').
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 window_size: int,
                 window_step: int,
                 anomaly_types_for_dict=None,
                 verbose: bool = True):
        if isinstance(dataset, Entity):
            dataset = Dataset(entities=[dataset], name=dataset.name, verbose=False)

        self.batch_size = batch_size
        self.window_size = window_size
        self.window_step = window_step
        self.anomaly_dict = Loader_aug._get_anomaly_dict(anomaly_types_for_dict or ['normal'])
        self.normal_label = t.Tensor(Loader_aug.generate_one_hot(['normal'], self.anomaly_dict))[0]

        # same entity filter and window positions as Loader_aug._array_to_windows
        self.arrays = []
        self.index = []
        for entity in dataset.entities:
            n_time = entity.Y.shape[1]
            if n_time >= window_size:
                k = len(self.arrays)
                self.arrays.append(np.ascontiguousarray(entity.Y, dtype=np.float32))
                starts = np.arange(0, n_time - window_size + 1, window_step)
                self.index.append(np.stack([np.full_like(starts, k), starts], axis=1))
        self.index = np.concatenate(self.index) if self.index else np.zeros((0, 2), dtype=int)
        self.n_idxs = len(self.index)
        self.n_batch_in_epochs = int(np.ceil(self.n_idxs / self.batch_size))

    def __len__(self):
        return self.n_idxs

    def __iter__(self):
        for b in range(self.n_batch_in_epochs):
            rows = self.index[b * self.batch_size:(b + 1) * self.batch_size]
            Y = t.from_numpy(np.stack([self.arrays[k][:, s:s + self.window_size] for k, s in rows]))
            yield {'Y': Y,
                   'Z': Y,
                   'anomaly_mask': t.ones_like(Y),
                   'label': self.normal_label.expand(len(rows), -1),
                   'idx': list(range(b * self.batch_size, b * self.batch_size + len(rows)))}
