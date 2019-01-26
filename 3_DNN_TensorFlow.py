##!/usr/bin/env python
'''
ABOUT:  This simple program intend to train the data of Dittus-Boelter 
        correaltions for the heat transfer (heating only). 
        The data were provided in a 'DittusBoelterDatabase.csv'. In addition,
        a MATLAB script is also provided to generate data beyond the 
        limits of Re and Pr considered here. 
        This specific program is to show the use of TensorFlow library along
        with hyperparameter tuning (using GridSearch).
        Keras library was used, which is a high level API for TensorFlow.
        Instead of matplotlib, seaborn is used here to demonstrate its usage.
        Its an high level API to matplotlib with rich options.
DEPENDS ON: Python3,XLRD, NumPy, Sklearn, matplotlib, Pandas, TensorFlow, Keras, seaborn
DATE:   24.01.2019
AUTHOR: Sandeep Pandey (sandeep.pandey@ike.uni-stuttgart.de)
LINCENSE: GPL-3.0
'''
print(__doc__)
# Import relavant modules here
import numpy as np
import pandas as pd
# Seed the random number for reproducability of the results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
# For profiling, use the timer
import time
start_time = time.time()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# User defined function to read, scale and split the data with some default value
def data_process(filename='DittusBoelterDatabase.csv',test_size=0.2):
    data = pd.read_csv(filename)
    # Split features and output
    X = data.iloc[:,:-1].values
    y = data.iloc[:,2].values
    # Scale the data (only X)
    scaler = StandardScaler()
    X_norm= scaler.fit_transform(X)
    # Split all data in training and testing
    X_train, X_test, y_train, y_test= train_test_split(X_norm,y,test_size=test_size,random_state=0)
    return X_train, X_test, y_train, y_test

# Read, process and split the data with the aid of "data_process()" function
X_train, X_test, y_train, y_test = data_process()

# Deep Neuaral Network with TensorFlow using Keras 
import keras
from keras.models import Sequential
from keras.layers  import Dense, Dropout

# # User defined function for abaseline model in the form of function with some default value
def baseline_model(optimizer='adam', kernel_initializer='normal',activation='relu', dropout_rate=0.0):
    # create model
    model= Sequential() 
    # Add hidden layer one by one || In first layer we have to define "input_dim"
    model.add(Dense(10, input_dim=2, kernel_initializer=kernel_initializer, activation=activation))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Add output layer
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor
modelMLP_TF = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
# Let's find the best hyperparameters using GridSearch. You can try with more parameters.
activation =  ['relu', 'selu', 'softmax', 'tanh']
kernel_initializer = ['normal']
optimizer = ['Adam', 'sgd']
epochs = [100, 200]
batch_size = [10, 100]

param_grid = dict( epochs=epochs, 
                  batch_size=batch_size, 
                  optimizer=optimizer, 
                  activation=activation,
                  kernel_initializer=kernel_initializer)

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=modelMLP_TF, param_grid=param_grid, verbose=2)
grid_search = grid.fit(X_train, y_train) 


print("-------Grid search to find the best hyperparameters is finished-------------\n")

# Now lets use the best hyperparameters for final DNN model
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
modelMLP = grid_search.best_estimator_
modelMLP.fit(X_train,y_train)
print("Best_parameters",best_parameters)

# Prediction test with the best parameter
y_MLP=modelMLP.predict(X_test)
from sklearn import metrics
print('Mean Absolute Error of DNN is:',metrics.mean_absolute_error(y_test,y_MLP))

# Plot the results with seaborn and tweak it with matplotlib
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
ax = sns.lineplot(y_test,y_test, color='red')
sns.scatterplot(y_test,y_MLP,  s=20, color='green', marker='o')
plt.xlabel('Nu$_{exact}$')
plt.ylabel('Nu$_{predicted}$')
plt.title('Comparision of actual value \nwith predicted value')
plt.show()

elapsed_time = (time.time()-start_time)/60.0
print("Total elapsed time for this code is=", "%.2f" %elapsed_time,"minutes")