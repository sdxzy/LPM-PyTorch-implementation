import torch.utils.data as data
import pickle
import torch
import numpy as np
class MyDataset(data.Dataset):
    def __init__(self, data_name='', train=True):
        ## init
        self.arch = []
        self.lat = []
        self.read_pickle_file(data_name)


    def __getitem__(self, idx):
        architecture = self.arch[idx]
        latency = self.lat[idx]
        arch_t = torch.from_numpy(np.asarray(architecture)).float()
        arch_l = torch.Tensor([latency]).float()
        return arch_t, arch_l


    def __len__(self):
        return len(self.lat)


    def read_pickle_file(self, data_name):
        with open(data_name, 'rb') as f:
            pair_data = pickle.load(f)
            f.close()
        self.get_data_pair(pair_data)

    def get_data_pair(self, pair_data):
        for key, value in pair_data.items():
            key_l = eval(key)
            self.arch.append(key_l)
            self.lat.append(value)
        assert len(self.lat) == len(self.arch)
