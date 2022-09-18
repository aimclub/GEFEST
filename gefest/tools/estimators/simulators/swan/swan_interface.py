import copy
import numpy as np
import subprocess

from gefest.core.structure.structure import Structure


class Swan:

    def __init__(self, path, targets, grid, domain):
        self.path_to_model = path
        self.path_to_input = path + 'INPUT'
        self.path_to_hs = path + 'r/hs47dd8b1c0d4447478fec6f956c7e32d9.d'
        self.targets = targets
        self.grid = grid
        self.domain = domain
        self._grid_configuration()

    def _grid_configuration(self):

        file_to_read = open(self.path_to_input, 'r')
        content_read = file_to_read.read()

        for_replace = ['xpc', 'xlenc', 'ypc', 'ylenc', 'mxc', 'myc']
        content_for_add = [self.domain.min_x,
                           self.domain.max_x,
                           self.domain.min_y,
                           self.domain.max_y,
                           self.grid[0],
                           self.grid[1]]

        new_content = copy.deepcopy(content_read)
        for replace, content in zip(for_replace, content_for_add):
            content_to_replace = replace + '=' + str(content)
            start = new_content.find(replace)
            end = new_content[start:].find(' ')
            content_write = new_content.replace(
                new_content[start:start + end], content_to_replace)
            new_content = content_write
        file_to_read.close()

        file_to_write = open(self.path_to_input, 'w')
        file_to_write.writelines(new_content)
        file_to_write.close()

    def estimate(self, struct: 'Structure'):
        polygons = struct.polygons

        file_to_read = open(self.path_to_input, 'r')
        content_read = file_to_read.read()

        for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
        num_of_polygons = len(polygons)
        for j, poly in enumerate(polygons):
            num_of_points = len(2 * poly.points)
            points = np.array([p.coords()[:2] for p in poly.points])
            individ = points.reshape(-1)
            for i, gen in enumerate(individ):
                if (i + 1) % 2 == 0:
                    if (i + 1) == num_of_points:
                        for_input += str(self.domain.max_y - gen)
                    else:
                        for_input += str(self.domain.max_y - gen) + ', '
                else:
                    for_input += str(gen) + ', '

            if j == (num_of_polygons - 1):
                for_input += '\n$optline'
            else:
                for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '

        content_to_replace = for_input
        content_write = content_read.replace(
            content_read[content_read.find('\n\n\n') + 3:content_read.rfind('\n$optline') + 9], content_to_replace)
        file_to_read.close()

        file_to_write = open(self.path_to_input, 'w')
        file_to_write.writelines(content_write)
        file_to_write.close()

        subprocess.run('swan.exe', shell=True, cwd=self.path_to_model)

        z = np.loadtxt(self.path_to_hs)
        hs_target = np.sum([z[target[0], target[1]] for target in self.targets])

        return z, hs_target
