import numpy as np 
from sympy import*
import pandas as pd



data = pd.read_csv('data/adult.data')


def data_processing(data):
    data = np.array(data)
    X = data[:,:-1]
    y = data[:,-1]
    return X,y

X,y = data_processing(data)
print(X.shape)