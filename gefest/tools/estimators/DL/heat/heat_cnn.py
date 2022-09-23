import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import PIL

from torch import nn


class HeatCNN:
    """
    ::TODO:: Make abstract version for creation own realizations for specific tasks
    """

    """
    Surrogate model for the heat components task
    """

    def __init__(self, path):
        super(HeatCNN, self).__init__()

        self.model = EffModel()
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device('cpu')))

        self.img_size = 128

    def estimate(self, obj):
        """
        Estimation step
        :param obj: (Tensor), [1 x C x W x H], object for estimate
        :return: (Float), performance of object
        """
        plt.imsave('tmp_images/0.png', torch.squeeze(torch.tensor(obj)))

        names = os.listdir('tmp_images')
        names = [os.path.splitext(name)[0] for name in names]
        tensors = torch.Tensor()
        with torch.no_grad():
            for name in names:
                image = PIL.Image.open(f'tmp_images/{name}.png').convert('L')
                tensor = transforms.ToTensor()(image)
                tensor = transforms.Resize([128, 128])(tensor)
                tensor = tensor.view(1, 1, 128, 128)
                tensors = torch.cat((tensors, tensor), dim=0)

            performance = self.model(tensors)[0].item()

        return performance


class EffModel(nn.Module):

    def __init__(self):
        super(EffModel, self).__init__()

        model = models.efficientnet_b0(pretrained=True)

        for param in model.parameters():
            param.requires_grad = True

        self.pre_model = nn.Sequential(*list(model.children())[:-2])

        self.preparation = nn.Conv2d(1, 3, kernel_size=3, padding='same')

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.mse = nn.L1Loss(reduction='mean')

    def forward(self, x):
        first = self.preparation(x)
        pre = self.pre_model(first)
        decoded = self.decoder(pre)

        return decoded

    def loss(self, pred, true):
        out = self.mse(torch.flatten(pred.float()), torch.flatten(true.float()))
        return out
