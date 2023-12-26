import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.tools.estimators.estimator import Estimator

matplotlib.use('agg')


class BWCNN(Estimator):
    """Surrogate model for breakwaters task."""

    def __init__(self, path, domain: Domain, main_model=None):
        super(BWCNN, self).__init__()

        self.domain = domain
        self.model = keras.models.load_model(path)
        self.main_model = main_model

        self._create_temp_path()
        self.img_name = 'tmp_images/0.png'
        self.img_size = 128
        self.rate = 4

    def _create_temp_path(self):
        """Creation of temporary folder for images."""
        path = 'tmp_images'

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

        return

    def _save_as_fig(self, struct: Structure, ax=plt):
        """Saves structs as images.

        Args:
            struct (Structure): _description_
            ax : Plot. Defaults to plt.
        """
        plt.style.use('dark_background')

        polygons = struct.polygons
        poly_area = self.domain.prohibited_area.polygons
        polygons = polygons + poly_area

        for poly in polygons:
            if poly.id == 'tmp':
                line_x = [point.x for point in poly.points]
                line_y = [point.y for point in poly.points]
                ax.plot(line_x, line_y, color='white', linewidth=3)
            elif poly.id == 'prohibited_area':
                line_x = [point.x for point in poly.points]
                line_y = [point.y for point in poly.points]
                ax.fill(line_x, line_y, color='white')

            elif poly.id == 'prohibited_poly' or 'prohibited_targets':
                line_x = [point.x for point in poly.points]
                line_y = [point.y for point in poly.points]
                ax.plot(line_x, line_y, color='white', linewidth=1)

        ax.axis('off')
        ax.axis(xmin=0, xmax=self.domain.max_x)
        ax.axis(ymin=0, ymax=self.domain.max_y)
        ax.savefig(self.img_name, bbox_inches='tight', pad_inches=0)
        ax.close('all')

    def _to_tensor(self, struct: Structure):
        """Transformation structure to binary tensor.

        Args:
            struct (Structure): Input structure

        Returns:
            Tensor: Binary matrix with WxHx1 dimension.
        """
        self._save_as_fig(struct)

        image_tensor = tf.io.read_file(self.img_name)
        image_tensor = tf.image.decode_png(image_tensor, channels=1)
        image_tensor = tf.image.resize(image_tensor, (self.img_size, self.img_size))
        image_tensor = image_tensor / 255

        return image_tensor

    def estimate(self, struct: Structure):
        """Estimation step.

        Args:
            struct (Structure), input structure.

        Returns:
            (float): Performance.
        """
        tensor = self._to_tensor(struct)
        tensor = tf.reshape(tensor, (1, self.img_size, self.img_size, 1))
        performance = self.model.predict(tensor)[0][0]

        if performance < self.rate:
            _, performance = self.main_model.estimate(struct)

        return performance
