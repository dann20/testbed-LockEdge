# When running this program with large data file, recommend executing on server
import pandas as pd
import time
import numpy as np
import pickle
NUM_HIDDEN_LAYER = 1
NUM_LAYERS = NUM_HIDDEN_LAYER + 1
# cols_index = 'B:D,F,H:Y,AN' # dropped flgs, state, seq, proto
cols_index = 'B,D,F,H,J:L,N,O,Q:AC,AR' # raw

# Load data
dir = r''
file = ['Label1', 'Label2', 'Label3', 'Label4', 'Label5', 'Label6', 
        'Label07_1', 'Label07_2', 'Label8', 'Label9', 'Label10', 'Label1-cut']
extension = '.xlsx'
file_index=-1
test = pd.read_excel(dir+file[file_index]+extension, usecols=cols_index)
test_labels = pd.DataFrame(test.iloc[:,-1])
testdata = test.drop(columns = 'attack')
# load_model = pickle.load(open('mlp_40.sav', 'rb'))

# Load MLP model
W = []
b = []
for i in range(NUM_LAYERS):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    k = k.reshape((k.shape[0],1))
    b.append(k)

# Normalize data
max = pickle.load(open('max_trainn.txt', 'rb'))
min = pickle.load(open('min_trainn.txt', 'rb'))
start = time.time()
for column in range(testdata.shape[1]):
    testdata.iloc[:,column] = testdata.iloc[:,column].apply(lambda x, column = column: (x-min[column])/(max[column]-min[column])) #RAM issue

# Test data
A = [testdata.to_numpy()]
for i in range(NUM_LAYERS):
    A.append(A[i].dot(W[i]) + b[i].reshape((b[i].shape[0],)))
    if (i != (NUM_LAYERS - 1)):
        A[-1] = np.maximum(0, A[-1]) 
print(time.time()-start)
test_labels = test_labels.to_numpy()
accuracy = np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_labels)
print(accuracy)
# print(load_model.score(testdata, test_labels))

# Export record to file
record = pd.DataFrame([[file[file_index]+extension, accuracy]], columns=['File', 'Accuracy'])
record.to_csv('mlp_accuracy.csv', mode = 'a', index=False, header=False)