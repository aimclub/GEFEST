import copy
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry import Structure
from gefest.tools import Estimator
from pathlib import Path

class Swan(Estimator):
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
        # self._grid_configuration()

    def _grid_configuration(self):

        file_to_read = open(self.path_to_input, 'r')
        content_read = file_to_read.read()

        for_replace = ['xpc', 'xlenc', 'ypc', 'ylenc', 'mxc', 'myc']
        content_for_add = [
            self.domain.min_x,
            self.domain.max_x,
            self.domain.min_y,
            self.domain.max_y,
            self.grid[0],
            self.grid[1],
        ]

        new_content = copy.deepcopy(content_read)
        for replace, content in zip(for_replace, content_for_add):
            content_to_replace = replace + '=' + str(content)
            start = new_content.find(replace)
            end = new_content[start:].find(' ')
            content_write = new_content.replace(
                new_content[start : start + end],
                content_to_replace,
            )
            new_content = content_write
        file_to_read.close()

        file_to_write = open(self.path_to_input, 'w')
        file_to_write.writelines(new_content)
        file_to_write.close()

    def estimate(self, struct: Structure):
        polygons = struct.polygons
        #
        file_toread = self.path_to_input+'_2'
        with open(file_toread, 'r') as file_to_read:
            content_read = file_to_read.read()
            file_to_read.close()


            # all_ = [Point(x=74.8, y=67.9174616), Point(x=74.8, y=67.9410473), Point(x=74.8192817, y=67.9410473),
            #        Point(x=74.8192817, y=67.9174616), Point(x=74.8, y=67.9174616)]
            # border = [[74.7, 74.84, 74.84, 74.7], [67.88, 67.88, 67.96, 67.96]]
            # plt.plot(border[0], border[1])
            # all_cd = [i.coords for i in all_]
            # for poly in polygons:
            #     poly_cd = [i.coords for i in poly]
            #
            #     plt.plot([x[0]/500 for x in poly_cd], [y[1]/500 for y in poly_cd])
            #     plt.plot([x[0] for x in all_cd], [y[1] for y in all_cd])
            #
            # plt.show()

            num_of_polygons = len(polygons)
            for j, poly in enumerate(polygons):
                for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
                num_of_points = len(2 * poly.points)
                points = np.array([p.coords[:2] for p in poly.points])
                individ = points.reshape(-1)
                for i, gen in enumerate(individ):
                    for_input += '{:.6f}'.format(gen) + ', '
                #for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '
                #if j == (num_of_polygons - 1):
            for_input += '\n$optline'
                # else:
                #     for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '

            content_to_replace = for_input
            content_write = content_read.replace(
                content_read[content_read.find('OBSTACLE') -1: content_read.rfind('$optline') + 10],
                content_to_replace,
            )
            #print(content_write)

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
        return z, hs_target
