import os

import cv2 as cv
import numpy as np
import torch
from tools.samplers.sampler import Sampler

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.tools.samplers.DL.microfluid.aae import AAE
from gefest.tools.samplers.DL.microfluid.backbones import (
    Decoder,
    Discriminator,
    Encoder,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DeepSampler(Sampler):
    """Deep learning sampler for microfluidic problem based on adversarial auto encoder.

    It is creates images of polygons with size 128x128.
    """

    def __init__(self, path):
        super(DeepSampler, self).__init__()

        self.path = path

        self.sampler = None
        self.device = None
        self.hidden_dim = None

        self._configurate_sampler()

    def _configurate_sampler(self):
        """Configurate deep sampler using configuration parameters."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conv_dims = [32, 64, 128, 256, 256, 512]  # 6 layers
        n_layers = 2
        self.hidden_dim = 32

        aae = AAE(
            Encoder=Encoder,
            Decoder=Decoder,
            Discriminator=Discriminator,
            hidden_dim=self.hidden_dim,
            conv_dims=conv_dims,
            n_layers=n_layers,
            device=self.device,
        )

        aae.load_state_dict(
            torch.load(self.path, map_location=self.device),
        )  # Load prepared sampler
        aae.eval()

        self.sampler = aae

    def _transform(self, objects, domain) -> list[Structure]:
        """Transforms images to polygons using edge detector.

        Args:
            objects (Array): [n_samples x 1 x 128 x 128].
            domain (Doamin): Task domain.

        Returns:
            List(Structure): Structures.
        """
        samples = []

        for obj in objects:
            im = np.squeeze(obj)
            ret, thresh = cv.threshold(im, 0.5, 1, 0)
            thresh = np.array(thresh).astype(np.uint8)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            polys = []
            for contour in contours:
                x = contour.reshape(-1, 2)[:, 0]
                y = contour.reshape(-1, 2)[:, 1]

                if np.min(x) == 0:
                    continue
                elif np.max(x) == 127:
                    continue

                x = np.array(x) / 128 * (domain.max_x - domain.min_x) - 128
                x = np.append(x, x[0])

                y = (np.array(y) / 128 * (domain.max_y - domain.min_y) - domain.max_y) * (-1)
                y = np.append(y, y[0])

                points = [Point(c1, c2) for c1, c2 in zip(x, y)]
                poly = Polygon(polygon_id='tmp', points=points)
                polys.append(poly)

            struct = Structure(polygons=polys)
            samples.append(struct)

        return samples

    def sample(self, n_samples: int, domain: Domain):
        """Sampling procedure using deep learning sampler."""
        with torch.no_grad():
            noise = torch.normal(mean=0, std=1, size=(n_samples, self.hidden_dim)).to(self.device)
            objects = self.sampler.decoder.sample(noise).numpy()  # Numpy: {n_samples, 1, 128, 128}

        # Obtained objects are images, we have to transform them to polygons
        samples = self._transform(objects, domain)

        return samples
