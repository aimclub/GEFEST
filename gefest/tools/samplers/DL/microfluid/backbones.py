import torch

from torch import nn


class Encoder(nn.Module):
    #####
    # Encoding input signal to sample of hidden conditional distribution q(z|x)
    #####
    def __init__(self, hidden_dim, conv_dims, device):
        """
        Constructing encoder
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param device: Current working device
        """
        super(Encoder, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.conv_dims = conv_dims

        model = []
        in_channel = 1  # Working with one dimensional input channel

        for channels in self.conv_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=channels,
                              kernel_size=3,
                              stride=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU())
            )
            in_channel = channels

        model.append(nn.AdaptiveMaxPool2d(output_size=1))
        model.append(nn.Flatten())

        self.model = nn.Sequential(*model)

        self.sample = nn.Linear(in_features=self.conv_dims[-1],
                                out_features=self.hidden_dim)

    def forward(self, x):
        """
        :param x: (Tensor) [B x C x W x H]
        :return: (Tensor) Tuple of expectation and logarithm of variance (dispersion)
        """
        conv_out = self.model(x)
        sample = self.sample(conv_out)

        return sample


class Decoder(nn.Module):
    ######
    # Decoding sample from conditional distribution q(z|x) to input like signal
    #####
    def __init__(self, hidden_dim, conv_dims, device):
        """
        Constructing decoder
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param device: Current working device
        """
        super(Decoder, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        conv_dims.reverse()  # Reversing dims to create decoder
        self.conv_dims = conv_dims  # It is possible to decrease number of parameters for prevent overfitting

        self.input_layer = nn.Linear(self.hidden_dim, self.conv_dims[0] * 4)

        """
        Each model`s layer will upscale input image size by two
        We are starting with 4x4 image`s size. Thus, we need
        five conv2transpose layers
        """
        model = []
        for i in range(len(self.conv_dims) - 1):
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.conv_dims[i],
                                       out_channels=self.conv_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1
                                       ),
                    nn.BatchNorm2d(self.conv_dims[i + 1]),
                    nn.ReLU()
                )
            )
        self.model = nn.Sequential(*model)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.conv_dims[-1],
                               out_channels=self.conv_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1
                               ),
            nn.BatchNorm2d(self.conv_dims[i + 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.conv_dims[-1],
                      out_channels=1,
                      kernel_size=3,
                      padding=1
                      ),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Decoding sample from latent distribution
        :param z: (Tensor) [B x hidden_dim]
        :return: (Tensor) [B x C x W x H]
        """
        input = self.input_layer(z).view(-1, self.conv_dims[0], 2, 2)
        out_decoder = self.model(input)
        out_final = self.final_layer(out_decoder)

        return out_final

    def sample(self, noise):
        """
        Decoding random noise to input like object
        :param noise: (Tensor) [B x hidden_dim]
        :return: (Tensor) [B x C x W x H]
        """
        assert noise.size(dim=0) != self.hidden_dim, 'Size of noise should equal hidden dimension'
        object_sample = self.forward(noise)

        return object_sample


class Discriminator(nn.Module):
    ######
    # Model for checking whether latent vector from true prior distribution or not
    #####
    def __init__(self, hidden_dim, n_layers, device):
        """
        Constructing discriminator with same structure as encoder
        :param hidden_dim: (Integer) Dimension of hidden vector
        :param device: Current working device
        """
        super(Discriminator, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        model = []

        in_dim = self.hidden_dim
        for _ in range(n_layers):
            model.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim,
                              out_features=in_dim * 4),
                    nn.Dropout(p=0.2),
                    nn.ReLU())
            )
            in_dim = in_dim * 4

        for _ in range(n_layers):
            model.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim,
                              out_features=int(in_dim / 4)),
                    nn.Dropout(p=0.2),
                    nn.ReLU())
            )
            in_dim = int(in_dim / 4)

        self.model = nn.Sequential(*model)

        self.prob = nn.Linear(in_features=in_dim,
                              out_features=1)

    def forward(self, x):
        """
        Deciding true or fake object
        :param x: (Tensor) [B x C x W x H]
        :return: (Float)
        """
        conv_out = self.model(x)
        prob = torch.sigmoid(self.prob(conv_out))

        return prob
