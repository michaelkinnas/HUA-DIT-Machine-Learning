import numpy as np

class LinearRegression:
    def __init__(self):
        w = None        # Weights
        b = None        # Intercept
        yhat = None     # Model predictions
        mse = None      # Mean squared error
        

    def fit(self, X, y):
        # Check if given parameters are o type ndarray
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError('Given parameters must be of type numpy array.')
        
        # Check if the given matrix parameter shapes are comatible for dot product.
        if X.shape[0] != y.shape[0]:
            raise ValueError('Matrices are wrong shape for matrix multiplication.')
     
        # Insert ones on last column of X
        X = np.insert(X, X.shape[1], 1, axis=1)

        # Calculate dot product
        res = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
       
        # Assign result to w and b class attributes.
        self.w = res[:-1]
        self.b = res[-1]
    
    def predict(self, X):
        # Check if w and b values are present.
        if self.w is None or self.b is None:
            raise ValueError('w and/or b are None. "fit" function must be run first.')
        
        # Check if given parameter is of type numpy array.
        if not isinstance(X, np.ndarray):
            raise TypeError('Given parameters must be of type numpy array.')
        
        return np.dot(X, self.w) + self.b
        
    def evaluate(self, X, y):
        # Check if w and b values are present.
        if self.w is None or self.b is None:
            raise ValueError('w and/or b are None. "fit" function must be run first.')
        
        # Check if given parameters are of type numpy array
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError('Given parameters must be of type numpy array.')
        
        # calculate yhat and MSE
        self.yhat = self.predict(X)
        self.mse = 1 / len(X) * np.dot((self.yhat - y).T, (self.yhat - y))

        return self.yhat, self.mse

    def print_vars(self):
        print(f'W: {self.w}')
        print(f'b: {self.b}')
        print(f'yhat: {self.yhat}')
        print(f'MSE: {self.mse}')
