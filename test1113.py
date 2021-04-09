#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:42:50 2020

@author: doanthang
"""

import pandas as pd
import numpy as np
import pickle
import subprocess
import time
from threading import Timer

NUM_HIDDEN_LAYER = 1
NUM_LAYERS = NUM_HIDDEN_LAYER + 1
# cols_index = 'B:D,F,H:Y,AN' #already dropped flgs, proto, state, seq
cols_index = 'B,D,F,H,J:L,N,O,Q:AC,AR'

def get_index(current_index):
    #global start_time
    #global dtime
    diff = time.time() - start_time
    step = 0
    while((current_index + step < len(dtime)) and (dtime[current_index + step] < diff)):
        step += 1
    if(step == 0):
        t = Timer((dtime[current_index] - diff), feed_forward, (current_index, current_index + 1))
        t.start()
    else:
        feed_forward(current_index, current_index + step)
    
def feed_forward(i, next_index):
    #global testdata
    ##global dtime
    #global test_labels
    #global start_time
    #global resmon_process
    test_batch = pd.DataFrame(testdata.iloc[i:next_index])
    for column in range(22):
        test_batch.iloc[:,column] = test_batch.iloc[:,column].apply(lambda x: (x-min[column])/(max[column]-min[column]))
    A = [test_batch.to_numpy()]
    for j in range(NUM_LAYERS):
        A.append(A[j].dot(W[j]) + b[j].reshape((b[j].shape[0],)))
        if (j != (NUM_LAYERS - 1)):
            A[-1] = np.maximum(0, A[-1]) 
    test_batch_labels = test_labels.iloc[i:next_index].to_numpy()
    print(np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_batch_labels), i, next_index)
    if(next_index==len(dtime)):
        dur = time.time() - start_time
        print(dur)
        resmon_process.terminate()
    else:
        get_index(next_index)

dir = r'/media/DATA/FIL/Dataset/UNSW_BOT/'
file = r'Label1-cut.xlsx'
test = pd.read_excel(dir+file, usecols=cols_index)
#test = pd.read_excel(dir+file)
testdata = test.drop(columns = 'attack')
testdata = testdata.append(testdata).append(testdata).append(testdata).append(testdata)
test_labels = pd.DataFrame(test.iloc[:,-1])
test_labels = test_labels.append(test_labels).append(test_labels).append(test_labels).append(test_labels)
stime = test.loc[:, 'stime']
dtime = stime - stime[0]
dtime = np.concatenate((dtime, dtime + 0.9186, dtime + 1.8372, dtime + 2.7558, dtime + 3.6744), 0)

sample_rate = 2400

dtime = dtime * 50000/sample_rate/0.9185
print(file)

W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)

max = pickle.load(open('max_trainn.txt', 'rb'))
min = pickle.load(open('min_trainn.txt', 'rb'))

resmon_process = subprocess.Popen(["resmon", "-o", "resource_1113_2400.csv"])
time.sleep(2)
start_time = time.time()
get_index(0)
