import numpy as np
import os 
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#path_to_dir = 'data_from_comsol//generated_data'

class SurrDataset(Dataset):
    def __init__(self, path_to_dir,end=None):
        self.path_to_dir = path_to_dir
        self.end = end
        self.flow = []
        self.mask = []
        self.__load_npz__()


    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        mask = self.mask[idx]
        flow = self.flow[idx]
        return mask,flow
    
    def __load_npz__(self):
        l_dir = os.listdir(self.path_to_dir)
        if self.end is not None:
            l_dir = l_dir[:self.end]
        for file in l_dir:
            mask = np.load(self.path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['mask']
            flow = np.load(self.path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow']
            for m,f in zip(mask,flow):
                self.flow.append(f)
                self.mask.append(m)

def create_single_dataloader(path_to_dir,
                       batch_size:int = 16,
                       shuffle=False):
    dataset = SurrDataset(path_to_dir=path_to_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)
    return train_loader

def create_dataloaders(path_to_dir,
                       batch_size:int = 16,
                       validation_split:float = 0.2,
                       shuffle_dataset:bool = True,
                       random_seed:int = 42,
                       end=None):
    """
    This function is return two dataloaders (train and test).
    it work for dataset, that consist of train and test data.(in my case i create only one folder with generated data
    and i whant to split Dataset but not folder with data)
    """
    
    dataset = SurrDataset(path_to_dir=path_to_dir,end=end)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader,test_loader
    

#train_loader,test_loader = create_dataloaders(path_to_dir,batch_size=32)
