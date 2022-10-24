import torch
import os
from gefest.tools.samplers.DL.microfluid.backbones import Encoder, Decoder, Discriminator
from gefest.tools.samplers.DL.microfluid.aae import AAE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DeepSampler:
    """
    Deep learning sampler for microfluidic problem based on adversarial auto encoder.
    It is creates images of polygons with size 128x128
    """

    def __init__(self, path):
        super(DeepSampler, self).__init__()

        self.path = path

        self.sampler = None
        self.device = None
        self.hidden_dim = None

        self._configurate_sampler()

    def _configurate_sampler(self):
        """
        Configurate deep sampler using configuration parameters
        :return: None
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conv_dims = [32, 64, 128, 256, 512, 512]  # We define 6 layers encoder and decoder
        n_layers = 2
        self.hidden_dim = 32

        aae = AAE(Encoder=Encoder,
                  Decoder=Decoder,
                  Discriminator=Discriminator,
                  hidden_dim=self.hidden_dim,
                  conv_dims=conv_dims,
                  n_layers=n_layers,
                  device=self.device)

        aae.load_state_dict(torch.load(self.path, map_location=self.device))  # Load prepared sampler
        aae.eval()

        self.sampler = aae

    def sample(self, n_samples: int, domain):
        """
        Sampling procedure using deep learning sampler.
        It based on general GEFEST deep learning sampler architecture,
        i.e. on mapping noise to object

        :param n_samples: (Int) number of samples
        :return: (List(Structure)) sample n_samples structures
        """
        with torch.no_grad():
            noise = torch.normal(mean=0, std=1, size=(n_samples, self.hidden_dim)).to(self.device)
            objects = self.sampler.decoder.sample(noise).to('cpu').tolist()  # Numpy: {n_samples, 1, 128, 128}

        return objects
