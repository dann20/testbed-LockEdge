import pandas as pd
import numpy as np
import pickle
NUM_HIDDEN_LAYER = 3
NUM_LAYERS = 4
cols_index = 'B:D,F,H:Y,AN'

file = r'C:\Users\shinn\OneDrive\FIL\Botnet_11102020\testset_daddr_6_raw.xlsx'
test = pd.read_excel(file, usecols=cols_index)
test_labels = pd.DataFrame(test.iloc[:,-1])
testdata = test.drop(columns = 'attack')

W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)

max = pickle.load(open('max_trainn.txt', 'rb'))
min = pickle.load(open('min_trainn.txt', 'rb'))

for row in range(len(testdata)):
    for column in range(22):
        testdata.iloc[row, column] = (testdata.iloc[row, column] - min[column])/(max[column]-min[column])
    A = [testdata.iloc[row].to_numpy()]
    for i in range(NUM_LAYERS):
        A.append(A[i].dot(W[i]) + b[i].reshape((b[i].shape[0],)))
        if (i != (NUM_LAYERS - 1)):
            A[-1] = np.maximum(0, A[-1]) 
    print(np.argmax(A[-1]), test_labels.loc[row,'attack'], 'True' if (np.argmax(A[-1]) == test_labels.loc[row,'attack']) else 'False')