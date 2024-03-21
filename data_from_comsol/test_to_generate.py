import os
import sys
import numpy as np
sys.path.append(os.getcwd())
print(os.getcwd())
sys.path.append(str(os.getcwd())+'/GEFEST/')
from gefest.core.geometry.datastructs.structure import Structure,Polygon,Point
from gefest.core.geometry.domain import Domain
from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol_generate
from cases.microfluidic.config_generate import opt_params
from gefest.tools.samplers.standard.standard import StandardSampler
from data_from_comsol.create_matrix import create_mask,create_flow
ITERATIONS = range(100)
SAMPLES = 10

comsol = Comsol_generate(path_to_mph='data_from_comsol//gen_setup.mph')
sampler = StandardSampler(opt_params=opt_params)

for i in ITERATIONS:
    data = {'mask':[],'flow':[],'struct':[]}
    structs = sampler.sample(n_samples=SAMPLES)
    for s in structs:
        results = comsol.estimate(structure = s)
        flow = create_flow(path='data_from_comsol//velocity.txt')
        mask = create_mask(path='data_from_comsol\mask.txt')
        data['flow'].append(flow)
        data['mask'].append(mask)
        data['struct'].append(s)
    np.savez(f'data_from_comsol/gen_data_extend/data__{i}',data,allow_pickle=False)
    print(f'Saved {(i+1)*SAMPLES} samples of simulation')
print()



