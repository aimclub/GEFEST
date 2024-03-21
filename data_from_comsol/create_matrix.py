import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import reconstruction
from scipy.ndimage.morphology import grey_dilation, generate_binary_structure, iterate_structure

def create_mask(path,dilation:bool=True):
    df=pd.read_csv(path,names=['x','y','color'],sep='\t')
    array = np.zeros((400,400))
    x = (np.array(df['x'])+200)
    y = (np.array(df['y'])+200)
    value = np.array(df['color'])
    array[np.intc(x),np.intc(y)]=value
    if dilation:
        array = grey_dilation(array,size=(2,2))
    return array

def create_flow(path,dilation:bool=True):
    df=pd.read_csv(path,names=['x','y','color'],sep='\t')
    array = np.ones((400,400))*-1
    x = (np.array(df['x'])+200)
    y = (np.array(df['y'])+200)
    value = np.array(df['color'])
    array[np.intc(x),np.intc(y)]=value
    if dilation:
        array = grey_dilation(array,size=(2,2))
    return array



if __name__=="__main__":
    df=pd.read_csv('data_from_comsol//velocity.txt',names=['x','y','color'],sep='\t')
    #df=pd.read_csv('data_from_comsol\mask.txt',names=['x','y','color'],sep='\t')

    x = (np.array(df['x'])+200)
    y = (np.array(df['y'])+200)
    value = np.array(df['color'])

    array_2 = np.ones((400,400))*-256
    array_2[np.intc(x),np.intc(y)]=value


    array_111 = grey_dilation(array_2,size=(2,2))
    plt.imshow(array_111,interpolation='nearest')
    plt.show()


    array = np.ones((400,400))*-256
    array[np.intc(x),np.intc(y)]=value
    # array[np.intc(x+0.25),np.intc(y+0.25)]=value
    # array[np.intc(x-0.25),np.intc(y-0.25)]=value

    # array[np.intc(x+0.25),np.intc(y-0.25)]=value
    # array[np.intc(x-0.25),np.intc(y+0.25)]=value
    plt.imshow(array,interpolation='nearest')
    plt.show()

    print()
