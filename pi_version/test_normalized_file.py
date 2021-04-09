import pandas as pd
import numpy as np
import time
NUM_HIDDEN_LAYER = 1
NUM_LAYERS = 2
cols_index = 'B:D,F,H:Y,AN'
file = ['testset_s_n_111020.xlsx','testset_daddr_3.xlsx','testset_daddr_6.xlsx','testset_daddr_67.xlsx','testset_daddr_567.xlsx']
j=0
test = pd.read_excel(file[j], usecols=cols_index)
test_labels = pd.DataFrame(test.iloc[:,-1])
testdata = test.drop(columns = 'attack')

W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)
start = time.time()
A = [testdata.to_numpy()]
for i in range(NUM_LAYERS):
    A.append(A[i].dot(W[i]) + b[i].reshape((b[i].shape[0],)))
    if (i!= (NUM_LAYERS-1)):
        A[-1] = np.maximum(0, A[-1])

dur = time.time() - start
print(dur)
test_labels = test_labels.to_numpy()
accuracy = np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_labels)
print(accuracy)
record = pd.DataFrame([[file[j], dur, accuracy]], columns=['File', 'Time','Accuracy'])
record.to_csv('mlp_test_record_40.csv', mode = 'a', index = False)
