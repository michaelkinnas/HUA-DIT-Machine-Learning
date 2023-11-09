import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression as mylr
from sklearn.linear_model import LinearRegression as sklr

# Prepare dataset
X, y = fetch_california_housing(data_home='data/', download_if_missing=True, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define linear regression models
mylr = mylr()        # My implementation of Linear Regression model
sklr = sklr()        # Scikit learn Linear Regression model

# Fit and evaluate on lr models
mylr.fit(X_train, y_train)
yhat_mylr, mse = mylr.evaluate(X_test, y_test)

sklr.fit(X_train, y_train)
yhat_sklr = sklr.predict(X_test)


# 3.1 Print root mean squred error
print(f'3.1)\nRoot Mean Squared Error: {np.sqrt(mse):.6f}\n')


# 3.2 Repeat for 20 times
rmses_mylr = []
rmses_sklr = []
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43+i)

    # Calculate for my linear regression model
    mylr.fit(X_train, y_train)
    pred, mse = mylr.evaluate(X_test, y_test)
    rmses_mylr.append(np.sqrt(mse))

    # Calculate for sklearn linear regression model
    sklr.fit(X_train, y_train)
    pred_sklr = sklr.predict(X_test)
    mse_sklr = 1 / len(X) * np.dot((pred_sklr - y_test).T, (pred_sklr - y_test))
    rmses_sklr.append(np.sqrt(mse_sklr))


# Print mean and std using pandas for my linear regression model
msedf = pd.DataFrame(rmses_mylr, columns = ["RMSE"])
print(f'3.2)\nMean value of RMSE: {msedf.mean().to_list()[0]:.6f}\nStandard deviation of RMSE: {msedf.std().to_list()[0]:.6f}\n')


# 3.3 Compare results with scikit learn linear regression model
preds_df = pd.DataFrame({
    'My LR model predictions': yhat_mylr,
    'Sklearn LR model predictions': yhat_sklr,
})

print(f'3.3)\nFirst 5 results\n{preds_df.head()}')