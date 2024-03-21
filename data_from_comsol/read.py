import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import reconstruction
from scipy.ndimage.morphology import grey_dilation, generate_binary_structure, iterate_structure

data = np.load('data_from_comsol\generated_data\data_1.npz',allow_pickle=True)

for i in data['arr_0'].item(0)['flow']:
    plt.imshow(i,interpolation='nearest')
    plt.show()