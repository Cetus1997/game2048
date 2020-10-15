# This is a data configuration
import torch
import numpy as np
from numpy import *
from torch.utils.data import Dataset,DataLoader
import copy
root = "./"

map_table = {2**i: i for i in range(1,16)}
map_table[0] = 0

def OneHotEncoder(InputArr):
     codearr = np.zeros((4,4,16),dtype=float)
     for p in range(0,4):
           for q in range(0,4):
                codearr[p,q,map_table[InputArr[p,q]]] = 1
     return codearr

class MyDataset(torch.utils.data.Dataset): 
    def __init__(self,txt, transform=None, target_transform=None): 
        fh = open(root + txt, 'r') 
        A = zeros((4,4),dtype = float)
        imgs = []
        lines = fh.readlines()
        row = 0
        for line in lines:
             if(row!=4):
                  list = line.strip('\n').split(' ')
                  A[row:] = list[0:4]
                  row = row+1
             else:
                  direction = int(line.strip('\n'))
                  row = 0
                  B = copy.deepcopy(A)
                  imgs.append((B,direction))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index] 
        arr = OneHotEncoder(fn)
        if self.transform is not None:
            arr = self.transform(arr)
        return arr,label
 
    def __len__(self): 
        return len(self.imgs)

