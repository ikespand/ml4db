# ML4DB
ML4DB acronymed for Machine Learning for Dittus-Boelter equation, and it is a simple machine learning program (for teaching purpose) written in Python. Dittus-Boelter equation is to find out the Nusselt Number which depends on Reynolds Number and Prandtl Numvber and type of heat transfer. [Exact solution](https://en.wikipedia.org/wiki/Nusselt_number#Dittus-Boelter_equation) is available for this equation.  

  - The database is derived from the [exact equation](https://en.wikipedia.org/wiki/Nusselt_number#Dittus-Boelter_equation) and provided in [DittusBoelterDatabase.csv](https://gitlab.com/spandey.ike/ml4db/blob/master/DittusBoelterDatabase.csv). One can regenerate the database with the attached [MATLAB script](https://gitlab.com/spandey.ike/ml4db/blob/master/generate_DB_data.m). 
  - Attached [Python script](https://gitlab.com/spandey.ike/ml4db/blob/master/train_db.py) can be used to read, train and test the [DittusBoelterDatabase.csv](https://gitlab.com/spandey.ike/ml4db/blob/master/DittusBoelterDatabase.csv).

### Dependencies
- Python3
- NumPy
- Sklearn
- Matplotlib
- Pandas
