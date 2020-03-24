import pandas as pd
import numpy as np 
data1 = pd.read_csv('dengue_features_train.csv')
data2 = pd.read_csv('dengue_labels_train.csv')
data3 = pd.read_csv('dengue_features_test.csv')
j=data1.corr()
y = data2.iloc[:,-1].values
X=pd.concat([data1.iloc[:, 0], data1.iloc[:, 4:]],axis=1).to_numpy()
#X = data1.iloc[:,4:].values

#Handling the missing data
#Imputer takes care o missing data in dataset

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
#To select columns from where we have to calculate data
imputer=imputer.fit(X[:,1:])
#To make changes to the original dataset
X[:,1:]=imputer.transform(X[:,1:])


#To deal with categorical data we had to convert it into numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
#converting first column into integer values
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#Encoding categorical data using one hot encoding
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#dividing the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Predicting a new result
'''X_test = pd.concat([data3.iloc[:, 0], data3.iloc[:, 4:]],axis=1).to_numpy()
imputer=imputer.fit(X_test[:,1:])
X_test[:,1:]=imputer.transform(X_test[:,1:])
X_test[:,0]=labelencoder_X.fit_transform(X_test[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X_test=onehotencoder.fit_transform(X_test).toarray()'''
y_pred = regressor.predict(X_test)
y_pred = np.round_(y_pred) 



