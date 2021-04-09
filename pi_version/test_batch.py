import pandas as pd
import numpy as np
import pickle
import subprocess
import time
NUM_HIDDEN_LAYER = 1
NUM_LAYERS = NUM_HIDDEN_LAYER + 1
# cols_index = 'B:D,F,H:Y,AN' #already dropped flgs, proto, state, seq
cols_index = 'B,D,F,H,J:L,N,O,Q:AC,AR'

def get_index(data, current_index, time_window):
    step = 1
    while ((step+current_index < len(data)) and ((data.loc[step+current_index, 'stime'] - data.loc[current_index, 'stime']) <= time_window)):
        step=step+1
    return step+current_index

dir = r'/home/pi/dataset_modified/testset_151020/'
file = r'Label1-cut.xlsx'
#test_file = pd.read_excel(dir+file, usecols=cols_index)
#test=pd.concat([test_file]*8)
#test=test.sort_values(by=['stime'], ascending=True, ignore_index = True)
test = pd.read_excel(dir+file, usecols=cols_index)
testdata = test.drop(columns = 'attack')
test_labels = pd.DataFrame(test.iloc[:,-1])
time_window = 0.1
print(file)

W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)

# min=[]
# max=[]
# for column in range(testdata.shape[1]):
#     max.append(testdata.iloc[:, column].max())
#     min.append(testdata.iloc[:, column].min())

max = pickle.load(open('max_trainn.txt', 'rb'))
min = pickle.load(open('min_trainn.txt', 'rb'))

resmon_process = subprocess.Popen(["resmon", "-o", "resources_50k_04.csv"])
time.sleep(2)
start = time.time()
i=0
while (i<len(testdata)):
    next_index = get_index(testdata, i,time_window )
    test_batch = pd.DataFrame(testdata.iloc[i:next_index])
    for column in range(22):
        test_batch.iloc[:,column] = test_batch.iloc[:,column].apply(lambda x: (x-min[column])/(max[column]-min[column]))
    A = [test_batch.to_numpy()]
    for j in range(NUM_LAYERS):
        A.append(A[j].dot(W[j]) + b[j].reshape((b[j].shape[0],)))
        if (j != (NUM_LAYERS - 1)):
            A[-1] = np.maximum(0, A[-1]) 
    if (i+1 == next_index):
        print(np.argmax(A[-1]), test_labels.loc[i,'attack'], 'True' if (np.argmax(A[-1]) == test_labels.loc[i,'attack']) else 'False', i)
    else:
        test_batch_labels = test_labels.iloc[i:next_index].to_numpy()
        print(np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_batch_labels), i, next_index)
    i = next_index
dur = time.time() - start
resmon_process.terminate()
print(dur)
record = pd.DataFrame([[file,dur,len(test),time_window]], columns = ['File','TestTime','NumberSamples','TimeWindow(s)'])
record.to_csv('mlp_test_batch_record_2810.csv', mode = 'a', index = False, header = False)

