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
from models import UNet,UNet_bi
import pandas as pd
from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.tensorboard import SummaryWriter

def main(args):
    makedirs(args)
    
    writer = SummaryWriter('writer/Loss')
    #snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = create_dataloaders(args.path_to_data,
                                                    batch_size = args.batch_size,
                                                    validation_split = 0.2,
                                                    shuffle_dataset = True,
                                                    random_seed = 42,
                                                    end=None)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet_bi(in_channels=1, out_channels=1)
    unet.to(device)

    ssim_loss = SSIM(data_range=2, size_average=True, channel=1)
    mae_loss = torch.nn.L1Loss()#torch.nn.MSELoss()
    best_validation_dsc = 0.0

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    sched =CosineLRScheduler(optimizer, t_initial=args.epochs-35, lr_min=0.0001,
                  cycle_mul=1.0, cycle_decay=1.0, cycle_limit=1,
                  warmup_t=3, warmup_lr_init=0.00005, warmup_prefix=False, t_in_epochs=True,
                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                  noise_seed=42, k_decay=1.0, initialize=True)
    #logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    mean_loss_train = []
    mean_loss_test = []

    step = 0
    ep = 0
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        writer.add_scalar('Lr',optimizer.param_groups[0]['lr'],ep)
        sched.step(ep)
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

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x,x)

                    loss = 0.5*(1-ssim_loss( y_pred, y_true.unsqueeze(1).float())) + mae_loss(y_pred.squeeze(), y_true)#loss = dsc_loss(y_pred.squeeze(), y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        writer.add_scalar('Loss/test',loss_valid[-1],step)
                        # y_pred_np = y_pred.detach().cpu().numpy()
                        # validation_pred.extend(
                        #     [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        # )
                        # y_true_np = y_true.detach().cpu().numpy()
                        # validation_true.extend(
                        #     [y_true_np[s] for s in range(y_true_np.shape[0])]
                        # )
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                        #         logger.image_list_summary(
                        #             tag,
                        #             log_images(x, y_true, y_pred)[:num_images],
                        #             step,
                        #         )

                    if phase == "train":
                        loss_train.append(loss.item())
                        writer.add_scalar('Loss/train',loss_train[-1],step)
                        loss.backward()
                        optimizer.step()

                if phase == "train":
                    mean_loss_train.append(np.mean(loss_train))
                    #mean_loss_test.append('')
                    print('train_loss',mean_loss_train[-1])
                    loss_train = []

            if phase == "valid":
                mean_loss_test.append(np.mean(loss_valid))
                print('test_loss',mean_loss_test[-1])
                torch.save(unet.state_dict(), os.path.join(args.weights, f"unet_{epoch}_bi.pt"))
                loss_valid = []
                result_train = pd.DataFrame(data = {'train_loss':mean_loss_train})
                result_test = pd.DataFrame(data = {'test_loss':mean_loss_test})
                result_test.to_csv('result_test_bi.csv')
                result_train.to_csv('result_train_bi.csv')
    #print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

# def snapshotargs(args):
#     args_file = os.path.join(args.logs, "args.json")
#     with open(args_file, "w") as fp:
#         json.dump(vars(args), fp)

# def log_loss_summary(logger, loss, step, prefix=""):
#     logger.scalar_summary(prefix + "loss", np.mean(loss), step)

# def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
#     dsc_list = []
#     num_slices = np.bincount([p[0] for p in patient_slice_index])
#     index = 0
#     for p in range(len(num_slices)):
#         y_pred = np.array(validation_pred[index : index + num_slices[p]])
#         y_true = np.array(validation_true[index : index + num_slices[p]])
#         dsc_list.append(dsc(y_pred, y_true))
#         index += num_slices[p]
#     return dsc_list


if __name__ == "__main__":
    from config import crete_parser
    args = crete_parser(batch_size=16,epochs=70,lr=0.1)
    main(args=args)
