import numpy as np 
from sympy import*
import pandas as pd



data = pd.read_csv('data/adult.data')


def data_processing(data):
    data = np.array(data)
    x = data[:,0]
    y = data[:,-1]
    
    return x,y

x,y = data_processing(data)