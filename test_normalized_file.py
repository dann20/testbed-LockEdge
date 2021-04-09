import pandas as pd
import numpy as np
import pickle
NUM_HIDDEN_LAYER = 3
NUM_LAYERS = NUM_HIDDEN_LAYER + 1
cols_index = 'B:D,F,H:Y,AN'
dir = r'C:\Users\shinn\OneDrive\FIL\Botnet_11102020\normalized_dataset'
test = pd.read_excel(dir+r'\testset_s_n_111020.xlsx', usecols=cols_index)
test_labels = pd.DataFrame(test.iloc[:,-1])
testdata = test.drop(columns = 'attack')
load_model = pickle.load(open('mlp_40_20_15.sav', 'rb'))

W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)

A = [testdata.to_numpy()]
for i in range(NUM_LAYERS):
    A.append(A[i].dot(W[i]) + b[i].reshape((b[i].shape[0],)))
    if (i != (NUM_LAYERS - 1)):
        A[-1] = np.maximum(0, A[-1]) 
test_labels = test_labels.to_numpy()
print(np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_labels))
print(load_model.score(testdata, test_labels))