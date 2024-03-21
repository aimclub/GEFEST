import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import create_single_dataloader
#from utils import log_images, dsc
from models import UNet
import pandas as pd


device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
dataloader = create_single_dataloader(path_to_dir='data_from_comsol//test_gen_data',batch_size=10,shuffle=False)
model = UNet(in_channels=1,out_channels=1).to(device)
model.load_state_dict(torch.load(f'weights//unet_58_mask_ssim_100.pt',map_location=torch.device(device)))
CASE = 'ssim_plus_58'
model.eval()
#predicts = []
truth = []
with torch.no_grad():
    print('start inference')
    for i, data in enumerate(dataloader):
        x, y_true = data
        x = x.to(device)

        y_pred = model(x,x).squeeze()
        if i==0:
            predicts = np.copy(y_pred.cpu().numpy())
            truth = np.copy(y_true)
            masks = np.copy(x.cpu().numpy())
        else:
            predicts = np.concatenate((predicts,y_pred.cpu().numpy()))
            truth = np.concatenate((truth,y_true))
            masks =  np.concatenate((masks,x.cpu().numpy()))
        #predicts+=y_pred.cpu().numpy()
    data_to_save = {'flow_predict':predicts,"flow":truth,'mask':masks}
    np.savez(f'gefest/surrogate_models/gendata/{CASE}/data',data_to_save,allow_pickle=False)
    print()
