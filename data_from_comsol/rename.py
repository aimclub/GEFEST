import os 
path = 'data_from_comsol\generated_data'
print(os.listdir(path))

for i , file in enumerate(os.listdir(path)):
    os.rename(path+f'/{file}',path+f'/data_gen_{i}.npz')