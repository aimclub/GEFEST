import json
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import create_dataloaders
#from utils import log_images, dsc
from models import AttU_Net, UNet,UNet_bi
import pandas as pd
from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion

def main(args,accum_gr):
    makedirs(args)
    
    writer = SummaryWriter('writer/Adam_acc')
    #snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = create_dataloaders(args.path_to_data,
                                                    batch_size = args.batch_size,
                                                    validation_split = 0.2,
                                                    shuffle_dataset = True,
                                                    random_seed = 42,
                                                    end=None)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = AttU_Net(img_ch=1,output_ch=1)#UNet_bi(in_channels=1, out_channels=1)
    unet.to(device)

    ssim_loss = SSIM(data_range=2, size_average=True, channel=1)
    mae_loss = torch.nn.L1Loss()#torch.nn.MSELoss()
    best_validation_dsc = 0.0

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)##Lion(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    sched =CosineLRScheduler(optimizer, t_initial=8, lr_min=0.00001,
                  cycle_mul=1.0, cycle_decay=1.0, cycle_limit=1,
                  warmup_t=0, warmup_lr_init=0.00001, warmup_prefix=False, t_in_epochs=True,
                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                  noise_seed=42, k_decay=1.0, initialize=True)
    #logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    mean_loss_train = []
    mean_loss_test = []
    accum_loss = []

    step = 0
    ep = 0
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        
        sched.step(ep)
        writer.add_scalar('Lr',optimizer.param_groups[0]['lr'],ep)
        ep+=1
        #print(optimizer.param_groups[0]['lr'])
        # for param_group in optimizer.param_groups:
        #         current_lr = param_group['lr']
        
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(tqdm(loaders[phase])):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x.float())

                    loss = 0.5*(1-ssim_loss( y_pred, y_true.unsqueeze(1).float())) + mae_loss(y_pred.squeeze(), y_true)#loss = dsc_loss(y_pred.squeeze(), y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        
                        
                    if phase == "train":
                        if accum_gr is not None:
                            loss_train.append(loss.item())
                            accum_loss.append(loss.item())
                            loss = loss/accum_gr
                            loss.backward()
                            if (i + 1) % accum_gr == 0:
                                writer.add_scalar('Loss/train',loss_train[-1],step)
                                
                                writer.add_scalar('Loss/accum_train',sum(accum_loss)/len(accum_loss),step)
                                optimizer.step()
                                optimizer.zero_grad()
                                accum_loss=[]
                        else:
                            loss_train.append(loss.item())
                            writer.add_scalar('Loss/train',loss_train[-1],step)
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                if phase == "train":
                    mean_loss_train.append(np.mean(loss_train))
                    #mean_loss_test.append('')
                    print('train_loss',mean_loss_train[-1])
                    loss_train = []

            if phase == "valid":
                mean_loss_test.append(np.mean(loss_valid))
                writer.add_scalar('Loss/test',mean_loss_test[-1],step)
                print('test_loss',mean_loss_test[-1])
                torch.save(unet.state_dict(), os.path.join(args.weights, f"unet_{epoch}_adam_Accum_2.pt"))
                loss_valid = []
                result_train = pd.DataFrame(data = {'train_loss':mean_loss_train})
                result_test = pd.DataFrame(data = {'test_loss':mean_loss_test})
                result_test.to_csv('result_test_adam_Accum_2.csv')
                result_train.to_csv('result_train_adam_Accum_2.csv')
    #print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)



if __name__ == "__main__":
    from config import crete_parser
    args = crete_parser(batch_size=7,epochs=70,lr=0.01)
    accumulatin_gradients = 10
    main(args=args,accum_gr=accumulatin_gradients)
