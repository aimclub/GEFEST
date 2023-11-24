import subprocess
from pathlib import Path

import numpy as np
from loguru import logger

from gefest.core.geometry import Structure
from gefest.tools import Estimator


class Swan(Estimator):
    """Class for SWAN estimator."""

    def __init__(
        self,
        path,
        targets,
        grid,
        domain,
        input_file_path='INPUT',
        hs_file_path='r/hs47dd8b1c0d4447478fec6f956c7e32d9.d',
    ):
        self.path_to_model = path
        self.path_to_input = path + input_file_path
        self.path_to_hs = path + hs_file_path
        self.targets = targets
        self.grid = grid
        self.domain = domain

    def estimate(self, struct: Structure) -> float:
        """Function to estimate wave high.

        :param struct: Structure with polygons

        :return: metric of wave high
        """
        polygons = struct.polygons

        file_toread = self.path_to_input + '_2'
        with open(file_toread, 'r') as file_to_read:
            content_read = file_to_read.read()
            file_to_read.close()
            for _j, poly in enumerate(polygons):
                for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
                points = np.array([p.coords[:2] for p in poly.points])
                individ = points.reshape(-1)
                for i, gen in enumerate(individ):
                    for_input += '{:.6f}'.format(gen / 500)
                    if i != len(individ) - 1:
                        for_input += ', '

            for_input += '\n$optline'
            content_to_replace = for_input
            content_write = content_read.replace(
                content_read[
                    content_read.find('OBSTACLE') - 1 : content_read.rfind('$optline') + 10
                ],
                content_to_replace,
            )

        input_created = Path(self.path_to_input)
        input_created.touch(exist_ok=True)
        with open(self.path_to_input, 'w') as file_to_write:
            file_to_write.write(content_write)

        logger.info('Swan estimation started...')
        subprocess.run(
            'swan.exe',
            shell=True,
            cwd=self.path_to_model,
            stdout=subprocess.DEVNULL,
        )
        logger.info('Swan estimation finished.')

        z = np.loadtxt(self.path_to_hs)
        res = []
        for i in range(1538 // 32):
            hs_target = np.sum(
                [z[i * 32 : (i + 1) * 32][target[0], target[1]] for target in self.targets],
            )

            res.append(hs_target)

        hs_target = sum(res) / len(res)
        print('hs_target', hs_target)
        return hs_target
