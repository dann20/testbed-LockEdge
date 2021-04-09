# normalize Botnet Dataset
import pandas as pd
import numpy as np

#Drop  flgs,proto,state,seq
cols_index = 'A,B,D,F:L,N:O,Q:AT'

# trainset = pd.read_excel('trainset_111020.xlsx', usecols=cols_index)
testset = pd.read_excel('testset_111020.xlsx', usecols=cols_index)

max=[]
min=[]

# dataset = [trainset, testset]
# file_names = ['trainset', 'testset']

dataset = [testset]
file_names = ['testset']

i = 0 # trainset 0, testset 1

# -3 due to ignoring 3 last label columns
for y in range(dataset[i].shape[1]-3):
    if ((y!=4) and (y!=6)): #ignore saddr and daddr
        max.append(dataset[i].iloc[:, y].max())
        min.append(dataset[i].iloc[:, y].min())
    else:
        max.append(0)
        min.append(0)

for y in range(dataset[i].shape[1]-3):
    if ((y!=4) and (y!=6)): 
        dataset[i].iloc[:,y] = dataset[i].iloc[:,y].apply(lambda x: (x-min[y])/(max[y]-min[y])) #RAM issue
    else:
        continue

dataset[i] = dataset[i].sample(frac=1).reset_index(drop=True) #shuffle data
dataset[i].to_excel(file_names[i]+'_s_n_111020.xlsx', index = False)

