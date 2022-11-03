import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class Movilens1MDataSet(Dataset):
    def __init__(self, dataset_name, dataset_path, device):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            df = pickle.load(f)
        self.dataset_name = dataset_name
        self.length = len(df)
        self.user_id = torch.from_numpy(np.array(list(df['user_id']))).reshape([self.length, -1]).to(device)
        self.item_id = torch.from_numpy(np.array(list(df['item_id']))).reshape([self.length, -1]).to(device)
        self.slate_id = torch.from_numpy(np.array(list(df['slate_id']))).reshape([self.length, -1]).to(device)
        self.slate_rating = torch.from_numpy(np.array(list(df['slate_rating']))).reshape([self.length, -1]).to(device)
        self.slate_pos = torch.from_numpy(np.array(list(df['slate_pos']))).reshape([self.length, -1]).to(device)
        self.rating = torch.from_numpy(np.array(list(df['rating']))).reshape([self.length, -1]).to(device)
    
    def __getitem__(self, index):
        feature = {
            'user_id': self.user_id[index],
            'item_id': self.item_id[index],
            'slate_id': self.slate_id[index],
            'slate_pos': self.slate_pos[index],
            'slate_rating': self.slate_rating[index]
        }
        return feature, self.rating[index]
    
    def __len__(self):
        return self.length

                        


