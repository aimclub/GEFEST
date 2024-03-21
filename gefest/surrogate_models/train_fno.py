import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import create_dataloaders
#from utils import log_images, dsc
from models import UNet
import pandas as pd
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from neuralop.models import TFNO

def main(args):
    makedirs(args)
    #snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = create_dataloaders(args.path_to_data,
                                                    batch_size = args.batch_size,
                                                    validation_split = 0.2,
                                                    shuffle_dataset = True,
                                                    random_seed = 42,
                                                    end=1)
    loaders = {"train": loader_train, "valid": loader_valid}

    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
    model.to(device)
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    dsc_loss = torch.nn.L1Loss()#torch.nn.MSELoss()
    best_validation_dsc = 0.0

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    mean_loss_train = []
    mean_loss_test = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x,x)

                    loss = (1-ssim_loss( y_pred, y_true.unsqueeze(1).float())) + dsc_loss(y_pred.squeeze(), y_true)
                    #ssim( y_pred, y_true.unsqueeze(1).float(), data_range=255, size_average=False)
                    if phase == "valid":
                        loss_valid.append(loss.item())

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
                torch.save(model.state_dict(), os.path.join(args.weights, f"fno_{epoch}_mask_ssim.pt"))
                loss_valid = []
                result_train = pd.DataFrame(data = {'train_loss_ssim':mean_loss_train})
                result_test = pd.DataFrame(data = {'test_loss_ssim':mean_loss_test})
                result_test.to_csv('result_test_mask_ssim.csv')
                result_train.to_csv('result_train_mask_ssim.csv')
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
    args = crete_parser(batch_size=16,epochs=25,lr=0.05)
    main(args=args)
