from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle
import numpy as np

file_train = r'C:\Users\shinn\OneDrive\FIL\Botnet_11102020\normalized_dataset\trainset_s_n_111020.xlsx'
file_test = r'C:\Users\shinn\OneDrive\FIL\Botnet_11102020\normalized_dataset\testset_s_n_111020.xlsx'
cols_index = 'B:D,F,H:Y,AN'

train = pd.read_excel(file_train, usecols=cols_index )
test = pd.read_excel(file_test, usecols=cols_index)
train_labels = pd.DataFrame(train.iloc[:,-1])
test_labels = pd.DataFrame(test.iloc[:,-1])
traindata = train.drop(columns = 'attack')
testdata = test.drop(columns = 'attack')

mlp = MLPClassifier(hidden_layer_sizes=(40,20), activation='relu', solver='adam', max_iter=500)
mlp.fit(traindata, train_labels)
print(mlp.score(testdata,test_labels))

file = 'mlp_40_20.sav'
pickle.dump(mlp, open(file,'wb'))

for i in range(len(mlp.coefs_)):
    np.savetxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),mlp.coefs_[i],delimiter=",")
    np.savetxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), mlp.intercepts_[i],delimiter=",")