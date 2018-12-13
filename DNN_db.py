##!/usr/bin/env python
'''
ABOUT:  This simple program intend to train the data of Dittus-Boelter 
        correaltions for the heat transfer (heating only). The data 
        were provided in a 'DittusBoelterDatabase.csv'. In addition, a 
        MATLAB script is also provided to generate data beyond the 
        limits of Re and Pr considered here. 
        The program is optimized for DNN.
DEPENDS ON: Python3,XLRD, NumPy, Sklearn, Matplotlib, Pandas
DATE:   26.10.2018
AUTHOR: Sandeep Pandey (sandeep.pandey@ike.uni-stuttgart.de)
LINCENSE: GPL-3.0
'''
print(__doc__)
# Import relavant modules here
import numpy as np
import pandas as pd
# Library to plot the data
import matplotlib.pyplot as plt 
#Predefine the properties of plot
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
# Import the necessary libraries for MachineLearning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
np.random.seed(10) # It is necessary to reproduce the results which depend on RANDOM NUMBERS

# Start of the program
print('Reading the data from the CSV file \n')
data=pd.read_csv('DittusBoelterDatabase.csv')

#Split features and output
X= data.iloc[:,:-1].values
y= data.iloc[:,2].values

#Split the all data in 80% (training) and 20% (testing)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)

print('Training the dataset \n')

# Neuaral network
scaler = StandardScaler() #this will normalize the data
scaler.fit(X_train)
X_train_norm= scaler.transform(X_train)
X_test_norm= scaler.transform(X_test)
modelMLP = MLPRegressor(hidden_layer_sizes=(2,6,1),
                        activation = 'tanh',
                        solver = 'adam',
                        learning_rate = 'constant',
                        learning_rate_init = 0.009,
                        batch_size = 16,
                        max_iter=1000)

modelMLP.fit(X_train_norm,y_train)

y_MLP=modelMLP.predict(X_test_norm)
print('Mean Absolute Error of ANN is:',metrics.mean_absolute_error(y_test,y_MLP))

# Plot the results
plt.plot(y_test,y_test,'k-', label='Ideal')
plt.plot(y_test,y_MLP,'r*', label='DNN')
plt.xlabel('Nu$_{exact}$')
plt.ylabel('Nu$_{predicted}$')
plt.title('Comparision of actual value \nwith predicted value')
plt.legend(loc="best")
ax = plt.gca()
ax.set_facecolor('#eafff5')
plt.show()
