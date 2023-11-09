import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression as lr
from sklearn.linear_model import LinearRegression as sklr

# Prepare dataset
X, y = fetch_california_housing(data_home='data/', download_if_missing=True, return_X_y = True)

# Create training split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define linear regression models
lr = lr()                               # My implementation of Linear Regressoin model

# Fit and evaluate on lr model
lr.fit(X_train, y_train)
yhat, mse = lr.evaluate(X_test, y_test)

# Fit and evaluate on sklearn lr model
sklr = sklr().fit(X_train, y_train)     # Scikit learn Linear Regression model
yhat_sklearn = sklr.predict(X_test)


# 3.1 Print root mean squred error
print(f'3.1) Root Mean Squared Error: {np.sqrt(mse)}')


# 3.2 Repeat for 20 times
rmses = []
rmses_skl = []
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42+i)

    lr.fit(X_train, y_train)
    pred, mse = lr.evaluate(X_test, y_test)
    rmses.append(np.sqrt(mse))

    sklr.fit(X_train, y_train)


# Print mean and std using pandas
msedf = pd.DataFrame(rmses, columns = ["RMSE"])
print(f'3.2) Mean value: {msedf.mean()}\nStandard deviation: {msedf.std()}')


# 3.3 Compare results with scikit learn linear regression model
print(yhat)
print(yhat_sklearn)