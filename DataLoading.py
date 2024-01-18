import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import numpy as np
import math


#Dataloader: Dataloader takes len getItem Constrcutor
#dataset: batchsize , dataloader , etc
#rest is the same for normal NN

class CarsData(Dataset):
    def __init__(self):
        self.data = np.loadtxt(r"C:\Users\Legend\Desktop\PyTorch\House Price Dataset.csv" , delimiter="," , skiprows=1 , dtype=np.float32 )
        self.x = torch.from_numpy(self.data[: , 1 : ])
        self.y = torch.from_numpy(self.data[: , : 1])
        self.n_samples = self.data.shape[0]
    def __getitem__(self, index):
        return self.x[index] , self.y[index]
    
    def __len__(self):
        return self.n_samples
dataset = CarsData()
dataloader = DataLoader(batch_size=4 , dataset=dataset)


dataiter = iter(dataloader)

data = next(dataiter)
print(data)


class LinearRegression(nn.Module):
    def __init__(self , input_size , output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size , output_size)