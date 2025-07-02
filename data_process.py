from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import torch
import os
import time


data_x = np.load('data/data_win_x.npy')
data_y = np.load('data/data_win_y.npy')


data_size = data_x.shape[0]

class ProcessedTemporalDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ProcessedTemporalDataset, self).__init__(root, transform, pre_transform)
        self.data_list = self.load_data()
        self.snapshot_count = len(self.data_list)

    def load_data(self):
        data_list = []
        for i in range(data_size):
            path = os.path.join(self.processed_dir, f'data_{i}.pt')
            data_list.append(torch.load(path))
        return data_list

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(data_size)]

    def process(self):

        source_nodes = torch.tensor([0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=torch.long)
        target_nodes = torch.tensor([1, 0, 3, 4, 5, 2, 4, 5, 2, 3, 5, 2, 3, 4], dtype=torch.long)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float)

        for i in range(data_x.shape[0]):
            x = torch.tensor(data_x[i].reshape(6, 1, 20), dtype=torch.float)

            y = torch.tensor(data_y[i].reshape(6, 1), dtype=torch.float)
            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            torch.save(snapshot, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


