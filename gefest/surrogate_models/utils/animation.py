import numpy as np
import os 
import sys
from matplotlib import pyplot as plt
import time
sys.path.append(os.getcwd())
print(os.getcwd())
sys.path.append(str(os.getcwd())+'/GEFEST/')
#path_to_dir = 'data_from_comsol//generated_data'

def animation_data_npz(path_to_dir):
    l_dir = os.listdir(path_to_dir)
    cnt = 0
    plt.ion()
    fig, axs = plt.subplots(1,2, figsize=(15, 10))
    summ_mae = 0
    for file in l_dir:
        mask = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['mask']#mask
        flow = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow']
        for m,f in zip(mask,flow):
            # cnt+=1
            # mse = np.square(f-np.where(m>-0.1,m,-1)).mean()
            # mae = np.sum(np.absolute(f-np.where(m>-0.1,m,-1)))/400/400
            # summ_mae+=np.sum(np.absolute(np.where(f>0,f,0)-np.where(m>-0.1,m,0)))
            m,f = np.where(m>-0.1,m,np.inf),np.where(f>0,f,np.inf)
            a1 = axs[0].imshow(m)
            
            # axs[1].set_title(f'summMAE: {summ_mae}')
            # axs[2].set_title(f'MAE: {mae},MSE: {mse}')
            a2 =axs[1].imshow(f)
             
            #a3 = axs[2].imshow(np.absolute(f-m),vmin = 0, vmax = 0.007)
            
            # fig.colorbar(a1,ax=axs[0])
            # fig.colorbar(a2,ax=axs[1])
            # fig.colorbar(a3,ax=axs[2])
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            axs[0].clear()
            axs[1].clear()
            #axs[2].clear()
            
            
            #time.sleep(1.2)
            a1.remove(),a2.remove()
        print(summ_mae)
        print(cnt)

def animation_npz(path_to_dir):
    l_dir = os.listdir(path_to_dir)
    cnt = 0
    plt.ion()
    fig, axs = plt.subplots(1,3, figsize=(15, 10))
    summ_mae = 0
    for file in l_dir:
        mask = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow_predict']#mask
        flow = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow']
        for m,f in zip(mask,flow):
            cnt+=1
            mse = np.square(f-np.where(m>-0.01,m,-1)).mean()
            mae = np.sum(np.absolute(f-np.where(m>-0.01,m,-1)))/400/400
            summ_mae+=np.sum(np.absolute(np.where(f>0,f,0)-np.where(m>-0.1,m,0)))
            m,f = np.where(m>-0.1,m,np.inf),np.where(f>0,f,np.inf)
            a1 = axs[0].imshow(m, vmin = 0, vmax = 0.05)
            
            axs[1].set_title(f'summMAE: {summ_mae}')
            axs[2].set_title(f'MAE: {mae},MSE: {mse}')
            a2 =axs[1].imshow(f, vmin = 0, vmax = 0.05)
             
            a3 = axs[2].imshow(np.absolute(f-m),vmin = 0, vmax = 0.007)
            
            # fig.colorbar(a1,ax=axs[0])
            # fig.colorbar(a2,ax=axs[1])
            # fig.colorbar(a3,ax=axs[2])
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
            
            
            time.sleep(2.2)
            a1.remove(),a2.remove(),a3.remove()
        print(summ_mae)
        print(cnt)


#animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim')
#animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim_23')
#animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim_plus_57')
#animation_data_npz(path_to_dir='data_from_comsol/gen_data_extend')
animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim_plus_58')