import numpy as np



def Objfun(y):
    
    return np.sum(np.abs(y)) + np.prod(np.abs(y))