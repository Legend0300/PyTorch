from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import numpy as np
import math
import torchvision


#Dataloader: Dataloader takes len getItem Constrcutor
#dataset: batchsize , dataloader , etc
#rest is the same for normal NN

class CarsData(Dataset):
    def __init__(self , transform=None):
        self.data = np.loadtxt(r"C:\Users\Legend\Desktop\PyTorch\House Price Dataset.csv" , delimiter="," , skiprows=1 , dtype=np.float32 )
        self.x = self.data[: , 1 : ]
        self.y = self.data[: , : 1]
        self.n_samples = self.data.shape[0]
        self.transform = transform
    def __getitem__(self, index):
        sample = self.x[index] , self.y[index]
        if self.transform:
          sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self , dataset):
        inputs , targets = dataset
        return torch.from_numpy(inputs) , torch.from_numpy(targets)
    

class Mult:
    def __init__(self , number):
        self.number = number

    def __call__(self , dataset):
        self.dataset = dataset
        inputs , targets = dataset
        inputs *= self.number
        targets *=self.number
        return inputs , targets
        
        

tranformations = torchvision.transforms.Compose([ToTensor() , Mult(2)])
dataset = CarsData(transform=tranformations)
dataloader = DataLoader(batch_size=4 , dataset=dataset)

test_data = dataset[0]
print(test_data)
# dataiter = iter(dataloader)

# data = next(dataiter)
# print(data)


class LinearRegression(nn.Module):
    def __init__(self , input_size , output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size , output_size)