# ML4DB
ML4DB acronymed for Machine Learning for Dittus-Boelter equation, and it is a simple machine learning program (for teaching purpose) written in Python. Dittus-Boelter equation is used to find out the Nusselt Number which depends on Reynolds Number and Prandtl Number based on the type of heat transfer. [Exact solution](https://en.wikipedia.org/wiki/Nusselt_number#Dittus-Boelter_equation) is available for this equation.  

  - The database is derived from the [exact equation](https://en.wikipedia.org/wiki/Nusselt_number#Dittus-Boelter_equation) and provided in [DittusBoelterDatabase.csv](https://github.com/ikespand/ml4db/blob/master/DittusBoelterDatabase.csv). One can regenerate the database with the attached [MATLAB script](https://github.com/ikespand/ml4db/blob/master/0_generate_DB_data.m). 
  - Attached first [Python script](https://github.com/ikespand/ml4db/blob/master/1_ML4DB_All.py) can be used to read, train and test the [DittusBoelterDatabase.csv](https://github.com/ikespand/ml4db/blob/master/DittusBoelterDatabase.csv). It shows the uses of different machine learning algorithm with some initial guess of hyperparameters.
  - Second [script](https://github.com/ikespand/ml4db/blob/master/2_DNN_DB.py) shows the implementation of deep neural network on the same database. Grid search, which is often used to find best hyperparameter, is also included.

### Dependencies
- Python3
- NumPy
- Sklearn
- Matplotlib
- Pandas
