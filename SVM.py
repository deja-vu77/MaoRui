Python 3.8.5 (v3.8.5:580fbb018f, Jul 20 2020, 12:11:27) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score #for accuracy calculation
from sklearn.svm import SVC
import time

for k in ['rbf','poly','sigmoid']:
    for i in np.arange(0.01,0.11,0.01):
        start=time.time()
        col_names = ['winner','gameId','creationTime', 'gameDuration', 'seasonId', 
                     'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                     'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
                     't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
                     't2_towerKills','t2_inhibitorKills','t2_baronKills',
                     't2_dragonKills','t2_dragonKills']

        # load dataset
        ts = pd.read_csv('test_set.csv')#training set
        ts = ts.iloc[1:] # delete the first row of the dataframe ts.head()

        nd = pd.read_csv('new_data.csv')#testing set
        nd = nd.iloc[1:] # delete the first row of the dataframe ts.head()



        #split dataset in features and target variable
        feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                        'firstDragon','firstRiftHerald','t1_towerKills',
                        't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                        't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                        't2_baronKills','t2_dragonKills','t2_dragonKills'] 

        X_train = ts[feature_cols] # Features
        y_train = ts.winner # Target variable

        X_test = nd[feature_cols] # Features
        y_test = nd.winner # Target variable



        # Create SVM classifer object 
        clf=SVC(kernel=k,gamma=i)
        # Train SVM Classifer
        clf=clf.fit(X_train,y_train)
        #Predict the response for test dataset 
        y_pred=clf.predict(X_test)
        # Model Accuracy
        SVM_A=accuracy_score(y_test,y_pred)

        end=time.time()
        print("kernal:",k)
        print("Accuracy:",accuracy_score(y_test, y_pred)) 
        print("Runing time: %s seconds"%(end-start))


