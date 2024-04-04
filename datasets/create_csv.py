import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

pth = 'currencyDataset/'
lst = os.listdir(pth)


path = []
tClass = []
for i in lst:
    for j in os.listdir(pth+i):
        path.append(pth+i+'/'+j)
        tClass.append(i)

d = {'pathname': path,'class': tClass}
df = pd.DataFrame(data=d)
df['mode'] = 'train'

df = df.sample(frac=1)
a = df.shape[0]
# 70% of dataset is train, 20% validation and 10% is test
a = int(a * 0.2)
df['mode'].iloc[:a] = 'val'
df['mode'].iloc[a:int(a*1.5)] = 'test'
df.to_csv('csv.csv')