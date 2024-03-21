import os
import sys
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(str(os.getcwd())+'/GEFEST/')
import matplotlib.pyplot as plt
from gefest.core.geometry.datastructs.structure import Structure,Polygon,Point
from gefest.core.geometry.domain import Domain
from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol_generate
from cases.microfluidic.config_generate import opt_params
from gefest.tools.samplers.standard.standard import StandardSampler
from data_from_comsol.create_matrix import create_mask,create_flow
ITERATIONS = range(10)
SAMPLES = 10

sampler = StandardSampler(opt_params=opt_params)

def plot_polygons(struct:Structure):

    for points in struct:
        x,y = [p.coords[0] for p in points],[p.coords[1] for p in points]
        plt.plot(x,y)
    

domain_points = opt_params.domain.allowed_area.points
x,y = [p.coords[0] for p in domain_points],[p.coords[1] for p in domain_points]
plt.plot(x,y)
for i in ITERATIONS:
    structs = sampler.sample(n_samples=SAMPLES)
    for s in structs:
        plot_polygons(s)
        print()
    plt.show()


    
    
print()



