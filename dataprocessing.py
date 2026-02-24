import numpy as np 
from sympy import*
import pandas as pd



data = pd.read_csv('data/adult.data')

pred = data.iloc[:,-1]
print(pred)

def data_processing(data):
    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    return x,y