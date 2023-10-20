import copy
import numpy as np
import subprocess

#from gefest.core.structure.structure import Structure


class Swan:

    def __init__(self, path, targets, grid, domain):
        self.path_to_model = path
        self.path_to_input = path + 'INPUT'
        self.path_to_hs = path + 'r/hs47dd8b1c0d4447478fec6f956c7e32d9.d'
        self.targets = targets
        self.grid = grid
        self.domain = domain
        self._grid_configuration()


    def estimate(self):


        file_to_read = open(self.path_to_input, 'r')
        content_read = file_to_read.read()

        for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '



        subprocess.run('swan.exe', shell=True, cwd=self.path_to_model)

        z = np.loadtxt(self.path_to_hs)
        hs_target = np.sum([z[target[0], target[1]] for target in self.targets])

        return z, hs_target