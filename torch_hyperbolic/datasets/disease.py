from torch_geometric.data import Data, InMemoryDataset, download_url
import torch
import os
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import train_test_split

class DiseaseDataset(InMemoryDataset):
    def __init__(self, root="./data/", holdout_size=0.6, seed=None, use_feats=True, *args, **kwargs):
        self.use_feats = use_feats
        self.holdout_size = holdout_size
        self.seed = seed
        super().__init__(root=root, *args, **kwargs)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["disease_nc.edges.csv", "disease_nc.feats.npz", "disease_nc.labels.npy"]
    
    @property
    def download_file_names(self):
        return ["disease_nc.edges.csv", "disease_nc.feats.npz", "disease_nc.labels.npy"]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "disease_raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "disease_processed")

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        base_path = "https://raw.githubusercontent.com/HazyResearch/hgcn/master/data/disease_nc/"
        [download_url(os.path.join(base_path, url), self.raw_dir) for url in self.download_file_names]

    def process(self):
        object_to_idx = {}
        idx_counter = 0
        edge_index = []
        with open(os.path.join(self.raw_dir, "disease_nc.edges.csv"), 'r') as f:
            all_edges = f.readlines()
        for line in all_edges:
            n1, n2 = line.rstrip().split(',')
            if n1 in object_to_idx:
                i = object_to_idx[n1]
            else:
                i = idx_counter
                object_to_idx[n1] = i
                idx_counter += 1
            if n2 in object_to_idx:
                j = object_to_idx[n2]
            else:
                j = idx_counter
                object_to_idx[n2] = j
                idx_counter += 1
            edge_index.append((i, j))

        y = np.load(os.path.join(self.raw_dir, "disease_nc.labels.npy"))
        
        if self.use_feats:
            x = torch.from_numpy(sp.load_npz(os.path.join(self.raw_dir, "disease_nc.feats.npz")).toarray())
        else:
            x = torch.eye(y.shape[0])
        y = np.load(os.path.join(self.raw_dir, "disease_nc.labels.npy"))

        indices = np.array(range(len(y)))
        if self.holdout_size > 0:
            train_indices, holdout_indices = train_test_split(
                indices, test_size=self.holdout_size, stratify=y)
        else:
            train_indices = indices

        train_mask = np.array([index in train_indices for index in indices], dtype=np.bool8)
        holdout_mask = ~train_mask

        try:
            if self.holdout_size > 0:
                test_indices, val_indices = train_test_split(
                indices[holdout_mask], test_size=0.5, stratify=y[holdout_mask])
            else:
                test_indices, val_indices = [], []
            test_mask = np.array([index in test_indices for index in indices])
            val_mask = np.array([index in val_indices for index in indices])
        except ValueError:
            train_mask = np.ones_like(train_mask).astype(np.bool8)
            val_mask = np.zeros_like(train_mask).astype(np.bool8)
            test_mask = np.zeros_like(train_mask).astype(np.bool8)

        assert np.sum(train_mask + val_mask + test_mask) == len(indices)

        data = Data(x=x.double(), 
                    y=torch.from_numpy(y).long(),
                    edge_index=torch.LongTensor(edge_index).t(),
                    train_mask=torch.BoolTensor(train_mask),
                    test_mask=torch.BoolTensor(test_mask),
                    val_mask=torch.BoolTensor(val_mask))
        torch.save(data, self.processed_paths[0])
    
