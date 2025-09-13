# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:33:45 2022

@author: 20182460
"""

import numpy as np 
import random
from random import randrange
from analytical_solution import calc_solution

data_dict = {}
for i in range(500):
    
    d1 = round(random.uniform(0.012, 0.035), 4)
    if d1 >= 0.0235:
        m = round(random.uniform(0.0009, 0.0093), 4)
        c = round(random.uniform(254.1e9, 507.8e9), 1)
        x0 = randrange(9243e9, 30209e9)
    else:
        m = round(random.uniform(0.0093, 0.0177), 4)
        c = round(random.uniform(0.4e9, 254.1e9), 1)
        x0 = randrange(3034e9, 9243e9)
        
    d2 = round(random.uniform(0.002, 0.047), 4)
    y0 = randrange(338.5e9, 674e9)
    
    data_t, data_x, data_y = calc_solution(x0, y0, d1, d2, m, c)
    data_dict[i] =  [data_t, data_x, data_y, d1, d2, m, c, x0, y0]

np.save('data_sets/data_no_noise', data_dict)

#add noise levels of 1%, 3% and 6% to original data 
noise_levels = [0.01, 0.05, 0.1]
def gaussian_noise(x, noise_level):
    std = noise_level*np.std(x)
    noise = np.random.normal(0.0, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 

data_dict_noise1 = {}
data_dict_noise5 = {}
data_dict_noise10 = {}
data_dicts_noise = [data_dict_noise1, data_dict_noise5, data_dict_noise10]

for j in range(len(noise_levels)):
    noise_level = noise_levels[j]
    data_dict_noise = data_dicts_noise[j]
    for i in range(len(data_dict)):
    
        tissue_count = data_dict[i][1]
        blood_count = data_dict[i][2]
        
        new_tissue_count = gaussian_noise(tissue_count, noise_level)
        new_blood_count = gaussian_noise(blood_count, noise_level)
        
        data_t = data_dict[i][0]
        d1 = data_dict[i][3]
        d2 = data_dict[i][4]
        m = data_dict[i][5]
        c = data_dict[i][6]
        x0 = data_dict[i][7]
        y0 = data_dict[i][8]
    
        data_dict_noise[i] = [data_t, new_tissue_count, new_blood_count, d1, d2, m, c, x0, y0]
    
np.save('data_sets/data_noise_1%', data_dict_noise1)
np.save('data_sets/data_noise_5%', data_dict_noise5)
np.save('data_sets/data_noise_10%', data_dict_noise10)
    
# data_0 = np.load('data_no_noise.npy', allow_pickle = True).item()
# data_1 = np.load('data_noise_1%.npy', allow_pickle = True).item()
# data_5= np.load('data_noise_5%.npy', allow_pickle = True).item()
# data_10 = np.load('data_noise_10%.npy', allow_pickle = True).item()

    




    
