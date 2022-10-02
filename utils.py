# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:26:41 2022

@author: ardit
"""

import numpy as np
from scipy.special import erfcinv 
from scipy import special as sp
# for PEDESTAL
# from https://matplotlib.org/stable/gallery/statistics/customized_violin.html
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

# for WAVEFORM
def MAD(df):
    c=-1/(np.sqrt(2)*erfcinv(3/2))
    df=np.abs(df-df.agg('median'))
    return c*(df.agg('median'))

def count_o(v):
    return np.size(v)-np.count_nonzero(v)

# for TRANSFER
def fit_linear(x,g0,g1):
    return x*g0+g1
def fit_cubic(x,g0,g1,g2,g3):
    return g0*x**3+g1*x**2+g2*x+g3

# for THRESHOLD
def err_func(x,a,b):
    
    return 0.5+0.5*(sp.erf((x-a)/((np.sqrt(2))*b)));