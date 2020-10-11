Python 3.8.5 (v3.8.5:580fbb018f, Jul 20 2020, 12:11:27) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>>import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score #for accuracy calculation
import matplotlib.pyplot as plt
import time

for i in range(10):
    start=time.time()

    col_names = ['winner','firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                 'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
                 't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
                 't2_towerKills','t2_inhibitorKills','t2_baronKills',
                 't2_dragonKills','t2_dragonKills']

    # load dataset
    ts = pd.read_csv('test_set.csv')#training set
    nd = pd.read_csv('new_data.csv')#testing set

    #Remap values from the winner column to numbers 0, 1.
    mappings = {
        1:0,
        2:1
    }
    ts['winner'] = ts['winner'].apply(lambda x: mappings[x])
    nd['winner'] = nd['winner'].apply(lambda x: mappings[x])


    #split dataset in features and target variable
    feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                    'firstDragon','firstRiftHerald','t1_towerKills',
                    't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                    't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                    't2_baronKills','t2_dragonKills','t2_dragonKills'] 

    X_train = ts[feature_cols].values # Features
    y_train = ts.winner.values # Target variable

    X_test = nd[feature_cols].values # Features
    y_test = nd.winner.values # Target variable


    #Convert split data from Numpy arrays to PyTorch tensors
    X_train = torch.FloatTensor(X_train) 
    X_test = torch.FloatTensor(X_test) 
    y_train = torch.LongTensor(y_train) 
    y_test = torch.LongTensor(y_test)


    #ANN model declaration
    class ANN(nn.Module): 
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features=16, out_features=100) 
            self.output = nn.Linear(in_features=100, out_features=2)
        def forward(self, x):
            x = torch.sigmoid(self.fc1(x)) 
            x = self.output(x)
            x = F.softmax(x,dim=1)
            return x

    model = ANN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



    
    #Model Training
    epochs = 550
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(X_train) 
        loss = criterion(y_hat, y_train) 
        loss_arr.append(loss)
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
    
    #Apply the model in the test-set.   
    predict_out = model(X_test)
    _,predict_y = torch.max(predict_out, 1)


    end=time.time()
    print("Epochs:",epochs)
    print("Accuracy:", accuracy_score(y_test, predict_y) )
    print("Runing time: %s seconds\n"%(end-start))