##!/usr/bin/env python
'''
ABOUT:  This simple program intend to train the data of Dittus-Boelter 
        correaltions for the heat transfer (heating only). 
        The data 
        were provided in a 'DittusBoelterDatabase.csv'. In addition, a 
        MATLAB script is also provided to generate data beyond the 
        limits of Re and Pr considered here. 
        This specific program is to show how hyperparameters can influence the performance and how to find them.
        This program can take upto 30-60 minutes to run on a single core.
DEPENDS ON: Python3,XLRD, NumPy, Sklearn, Matplotlib, Pandas
DATE:   11.07.2019
AUTHOR: Sandeep Pandey (sandeep.pandey@ike.uni-stuttgart.de)
LINCENSE: GPL-3.0
'''
print(__doc__)
# Import relavant modules here
import numpy as np
import pandas as pd
# Library to plot the data
import matplotlib.pyplot as plt 
# Predefine the properties of plot
params = {
    'axes.labelsize': 22,
    'font.size': 22,
    'font.family': 'Times New Roman',
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': False,
    'figure.figsize': [7, 5],
    'lines.linewidth': 4,
    'lines.markersize':2
}
plt.rcParams.update(params)
np.random.seed(10) # It is necessary to reproduce the results which depend on RANDOM NUMBERS

# Start of the program
data = pd.read_csv('DittusBoelterDatabase.csv')

# Split features and output
X = data.iloc[:,:-1].values
y = data.iloc[:,2].values

# Split all data in 80% (training) and 20% (testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #this will normalize the data
scaler.fit(X_train)
X_train_norm= scaler.transform(X_train)
X_test_norm= scaler.transform(X_test)

# Deep Neuaral Network
from sklearn.neural_network import MLPRegressor
modelMLP = MLPRegressor(max_iter=1000,
                        tol=0.0001,
                        batch_size='auto',
                        random_state=0)

# Perfrom grid search to find the best hyperparameters
from sklearn.model_selection import GridSearchCV
parameters = [{'hidden_layer_sizes':[(25, 25, 25, 25), (10, 10, 10, 10)],
               'activation' : [ 'logistic', 'tanh', 'relu'],
               'solver' : ['lbfgs', 'adam'],
               'alpha' : [0.0001, 0.0012],
               'learning_rate_init' : [0.0001, 0.0002]}]

grid_search = GridSearchCV(estimator = modelMLP,
                          param_grid = parameters,
                          scoring='neg_mean_absolute_error',
                          verbose=10,
                          n_jobs=1)

grid_search = grid_search.fit(X_train_norm,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("-------Grid search to find the best hyperparameters is finished-------------")

# Now lets use the best hyperparameters for final DNN model
modelMLP1 = grid_search.best_estimator_
modelMLP1.fit(X_train_norm,y_train)

#---Prediction test---
y_MLP1=modelMLP1.predict(X_test_norm)
from sklearn import metrics
print('Mean Absolute Error of ANN is:',metrics.mean_absolute_error(y_test,y_MLP1))

# Plot the results
plt.plot(y_test,y_test,'k-', label='Ideal')
plt.plot(y_test,y_MLP1,'r*', label='DNN')
plt.xlabel('Nu$_{exact}$')
plt.ylabel('Nu$_{predicted}$')
plt.title('Comparision of actual value \nwith predicted value')
plt.legend(loc="best")
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
plt.show()