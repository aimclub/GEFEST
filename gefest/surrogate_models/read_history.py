import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

step = 100
df = pd.read_csv('result_train_mask_ssim.csv')
df_test = pd.read_csv('result_test_mask_ssim.csv')
bins = [df['train_loss'][i:i+step].mean() for i in range(0,len((df['train_loss'])),step)]
#mean = df['train_loss'].groupby(pd.cut(df['train_loss'],bins))
plt.plot(df_test['test_loss'],label='base_test')
plt.plot(bins,label='base_train')
#plt.plot([0.013666182630779233,0.00986096077757238,0.006160223454961358,0.00486703837721719,0.005280730882590731,0.0028021071640542045])

df = pd.read_csv('result_train_bi.csv')
df_test = pd.read_csv('result_test_bi.csv')
bins = [df['train_loss'][i:i+step].mean() for i in range(0,len((df['train_loss'])),step)]
plt.plot(df_test['test_loss'],label='ssim_test')
plt.plot(bins,label='ssim_test')
plt.grid()
plt.legend()
plt.show()