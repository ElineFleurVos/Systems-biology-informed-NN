# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 23:59:54 2022

@author: 20182460
"""
import numpy as np 
from math import exp
import matplotlib.pyplot as plt
import random
from random import randrange

def calc_solution(x0, y0, d1, d2, m, c, t_max = 500):
    
    alpha = (m + d1)
    Cx = (d1*c)/(m+d1)
    Cy = (m*Cx)/d2

    data_t = np.empty(t_max, dtype=float) 
    data_x = np.empty(t_max, dtype=float) 
    data_y = np.empty(t_max, dtype=float)
    
    np.empty([2, 2], dtype=int)

    
    for t in range(t_max):
        x = Cx + (x0 - Cx)*exp(-alpha*t)

        if d2-alpha != 0:
            y_p1 = ( (m*x0)/(d2-alpha) - (m*d1*c)/(alpha*(d2-alpha)) )*exp(-alpha*t)
            y_p2 = (y0 - (m*d1*c)/(alpha*d2) - (m*x0)/(d2-alpha) + (m*d1*c)/((d2-alpha)*alpha) )*exp(-d2*t)
        else: 
            y_p1 = ( (m*x0)/(0.00001) - (m*d1*c)/(alpha*(0.00001)) )*exp(-alpha*t)
            y_p2 = (y0 - (m*d1*c)/(alpha*d2) - (m*x0)/(0.00001) + (m*d1*c)/((0.00001)*alpha) )*exp(-d2*t)
            
        y = Cy + y_p1 + y_p2 
       
        data_t[t] = t
        data_x[t] = x
        data_y[t] = y 
       
    return data_t, data_x, data_y









