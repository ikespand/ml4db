##!/usr/bin/env python
'''
ABOUT:  This simple program intend to train the data of Dittus-Boelter 
        correaltions for the heat transfer (heating only). The data 
        were provided in a 'DittusBoelterDatabase.csv'. In addition, a 
        MATLAB script is also provided to generate data beyond the 
        limits of Re and Pr considered here. 
        The program is not optimized for any parameters.
DEPENDS ON: Python3, NumPy, Sklearn, Matplotlib, Pandas
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
    'axes.labelsize' : 22,
    'font.size' : 22,
    'font.family' : 'Times New Roman',
    'legend.fontsize' : 18,
    'xtick.labelsize' : 18,
    'ytick.labelsize' : 18,
    'text.usetex' : False,
    'figure.figsize' : [7, 5],
    'lines.linewidth' : 4,
    'lines.markersize' :2
}
plt.rcParams.update(params)
np.random.seed(10) # It is necessary to reproduce the results which depend on RANDOM NUMBERS

# Start of the program
data = pd.read_csv('DittusBoelterDatabase.csv')

# Split features and output
X = data.iloc[:,:-1].values
y = data.iloc[:,2].values

# Split the all data in 80% (training) and 20% (testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#---------Train the network---------
# 1. Linear regression
from sklearn.linear_model import LinearRegression
modelLin = LinearRegression()
modelLin.fit(X_train,y_train)

# 2. Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) 
# Mapping data to polynomials i.e. a and b for degree=2-> constant, a, b, a2, ab, b2  
X_train_poly = poly_reg.fit_transform(X_train) 
modelPoly = LinearRegression()
modelPoly.fit(X_train_poly, y_train)

# 3. Random forest
from sklearn.ensemble import RandomForestRegressor
modelRF=RandomForestRegressor(n_estimators=1000,random_state=42) 
modelRF.fit(X_train,y_train)

# 4. Support vector regression
# SVR and DNN requires the feature scaling, so we will do scaling
# One can also scale y, but I am not doing it here.
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
scalerX = StandardScaler() 
scalerX.fit(X_train)
X_train_norm = scalerX.transform(X_train)
X_test_norm = scalerX.transform(X_test)
modelSVR = SVR(kernel='rbf')
modelSVR.fit(X_train_norm, y_train)
 
# 5. Neuaral network, with some initial guess
from sklearn.neural_network import MLPRegressor
modelMLP = MLPRegressor(hidden_layer_sizes=(2,6,1),
                        activation = 'tanh',
                        solver = 'adam',
                        learning_rate = 'constant',
                        learning_rate_init = 0.009,
                        batch_size = 16,
                        max_iter = 1000)

modelMLP.fit(X_train_norm,y_train)


#---Prediction test---
from sklearn import metrics
y_Lin= modelLin.predict(X_test)
y_Poly=modelPoly.predict(poly_reg.fit_transform(X_test))
y_RF=modelRF.predict(X_test)
y_SVR= modelSVR.predict(X_test_norm)
y_MLP=modelMLP.predict(X_test_norm)


print('Mean Absolute Error of Linear Regression is:',metrics.mean_absolute_error(y_test,y_Lin))
print('Mean Absolute Error of Polynomial is:',metrics.mean_absolute_error(y_test,y_Poly))
print('Mean Absolute Error of Random Forest is:',metrics.mean_absolute_error(y_test,y_RF))
print('Mean Absolute Error of SVR is:',metrics.mean_absolute_error(y_test,y_SVR))
print('Mean Absolute Error of ANN is:',metrics.mean_absolute_error(y_test,y_MLP))

# Plot the results
plt.plot(y_test,y_test,'k-', label='Ideal')
plt.plot(y_test,y_Lin,'b*', label='LR')
plt.plot(y_test,y_Poly,'c*', label='Poly')
plt.plot(y_test,y_RF,'g*', label='RF')
plt.plot(y_test,y_SVR,'y*', label='SVR')
plt.plot(y_test,y_MLP,'r*', label='DNN')
plt.xlabel('Nu$_{exact}$')
plt.ylabel('Nu$_{predicted}$')
plt.title('Comparision of actual value \nwith predicted value')
plt.legend(loc="best")
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
plt.show()
