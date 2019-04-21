import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

class CustomSGDLinearRegressor:
    
    def __init__(self, learning_rate, iters, k_rand_points):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.r = learning_rate
        self.k = k_rand_points
        self.iters = iters
        
    def fit(self, DTrain, y):
        
        self.X = np.array(DTrain)
        self.y = np.array(y).reshape(-1)
        self.w = np.random.normal(size = self.X.shape[1])
        self.b = np.random.normal()
        
        for _ in range( self.iters // self.k + 1):
            self.findBestBias()
            self.findBestWeights()
    
    def findBestWeights(self):
        
        opt_weights = self.w
        r = self.r
        
        for _ in range(self.iters - 1):
            opt_weights = opt_weights + ( r * 2 * self.selectKPointsforWeights( opt_weights ) ) / self.X.shape[0]
            
            r /= 2
        
        self.w = opt_weights
            
    def selectKPointsforWeights(self, weights):
        
        idxs = np.random.randint(0, self.X.shape[0], size = self.k)
        
        return (self.X[idxs] * (self.y[idxs] - (weights * self.X[idxs]).sum(axis=1) - self.b).reshape(-1,1)).sum(axis=0)
        
    def findBestBias(self):
        
        opt_b = self.b
        r = self.r
        
        for _ in range(self.iters - 1):
            opt_b = opt_b + ( r * 2 * self.selectKPointsforBias( opt_b ) ) / self.X.shape[0]
                
            r /= 2
        
        self.b = opt_b
            
    def selectKPointsforBias(self, bias):
        
        idxs = np.random.randint(0, self.X.shape[0], size = self.k)
        
        return (self.y[idxs] - (self.w * self.X[idxs]).sum(axis=1) - bias).sum()
        
    def predict(self, DTest):
        
        DTest = np.array(DTest)
        
        return np.round( (self.w* DTest).sum(axis=1) + self.b, 1 )
