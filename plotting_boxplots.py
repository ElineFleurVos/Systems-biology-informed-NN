# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:42:47 2022

@author: 20182460
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#%% NOISE METHOD 1

folder_test = 'results/MAPEs_solution/m1_test/'
folder_train = 'results/MAPEs_solution/m1_train/'

m1_no_noise = np.load(folder_test + 'm1_no_noise_b1_t1.npy')
m1_noise1= np.load(folder_test + 'm1_noise1_b1_t1.npy')
m1_noise5 = np.load(folder_test + 'm1_noise5_b1_t1.npy')
m1_noise10= np.load(folder_test + 'm1_noise10_b1_t1.npy')

all_noise_results_test = [m1_no_noise, m1_noise1, m1_noise5, m1_noise10]

labels_noise = ['no noise', '1% noise', '5% noise', '10% noise']
plt.figure(1)
plt.boxplot(all_noise_results_test, vert=False, labels=labels_noise)
plt.xlabel('MAPE')


m1_no_noise_train = np.load(folder_train + 'm1_no_noise_b1_t1.npy')
m1_noise1_train = np.load(folder_train + 'm1_noise1_b1_t1.npy')
m1_noise5_train = np.load(folder_train + 'm1_noise5_b1_t1.npy')
m1_noise10_train = np.load(folder_train + 'm1_noise10_b1_t1.npy')
all_noise_results_train = [m1_no_noise_train, m1_noise1_train, m1_noise5_train, m1_noise10_train]

means_noise_test = []
for i in all_noise_results_test:
    means_noise_test.append(np.mean(i))
    
means_noise_train = []
for i in all_noise_results_train:
    means_noise_train.append(np.mean(i))

#%% SAMPLING STRATEGY METHOD 1

strategy_11 = np.load(folder_test + 'm1_no_noise_b1_t1.npy')
strategy_21 = np.load(folder_test + 'm1_no_noise_b2_t1.npy')
strategy_31 = np.load(folder_test + 'm1_no_noise_b3_t1.npy')
strategy_12 = np.load(folder_test + 'm1_no_noise_b1_t2.npy')
strategy_22 = np.load(folder_test + 'm1_no_noise_b2_t2.npy')
strategy_32 = np.load(folder_test + 'm1_no_noise_b3_t2.npy')
strategy_13 = np.load(folder_test + 'm1_no_noise_b1_t3.npy')

sb1 = [0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
sb2 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
sb3 = []
tp = 0
for i in range(52):
    sb3.append(tp)
    tp += 7

st1 = [0, 50]
st2 = [0]
st3 = [0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]

all_strategy_results = [strategy_11, strategy_21, strategy_31, strategy_12, strategy_22, strategy_32, strategy_13]
labels_strategy = ['', '', '', '', '', '','']
colors = ['b','g','r','c','m','y','k']

blood_strategies = [sb1, sb2, sb3, sb1, sb2, sb3, sb1]
tissue_strategies = [st1, st1, st1, st2, st2, st2, st3]

plt.figure(2, figsize=(10,4))
#plt.suptitle('Boxplots with MAPEs for different sampling strategies')
plt.subplot(1,2,1)
places_y = [20,17,14,11,8,5,2]
for i in range(7):
    color = colors[i]
    blood_strategy = blood_strategies[i]
    tissue_strategy = tissue_strategies[i]
    
    x_blood= np.array(blood_strategy)
    y_blood = np.array(len(blood_strategy)*[places_y[i]])
    x_tissue = np.array(tissue_strategy)
    y_tissue = np.array(len(tissue_strategy)*[places_y[i]-1])
    plt.plot(x_blood, y_blood, '.', color = color)
    plt.plot(x_tissue, y_tissue,'^', color = color)
plt.ylabel('sampling strategy', size = 13)
plt.xlabel('days after treatment', size = 13)
plt.tick_params(left = False, right = False , labelleft = False)

plt.subplot(1,2,2)
plt.boxplot(all_strategy_results, vert=False, labels = labels_strategy)
plt.xlabel('MAPE', size = 13)
plt.xlim(0,2500)

plt.subplots_adjust(wspace=0.05)
plt.show()

means_sampling_test = []
for i in all_strategy_results:
    means_sampling_test.append(np.mean(i))
    
#calculate means of training set

strategy_11_train = np.load(folder_train + 'm1_no_noise_b1_t1.npy')
strategy_21_train = np.load(folder_train + 'm1_no_noise_b2_t1.npy')
strategy_31_train = np.load(folder_train + 'm1_no_noise_b3_t1.npy')
strategy_12_train = np.load(folder_train + 'm1_no_noise_b1_t2.npy')
strategy_22_train = np.load(folder_train + 'm1_no_noise_b2_t2.npy')
strategy_32_train = np.load(folder_train + 'm1_no_noise_b3_t2.npy')
strategy_13_train = np.load(folder_train + 'm1_no_noise_b1_t3.npy')
all_strategy_results_train = [strategy_11_train, strategy_21_train, strategy_31_train, strategy_12_train
                           ,strategy_22_train, strategy_32_train, strategy_13_train]
means_sampling_train = []
for i in all_strategy_results_train:
    means_sampling_train.append(np.mean(i))
    
#%% NOISE METHOD 2
no_noise_m2 = np.reshape(np.load('results/MAPEs_solution/m2/m2_no_noise_s1.npy'), (10,))
noise_1_m2 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise1_s1.npy'),(10,))
noise_5_m2 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise5_s1.npy'),(10,))
noise_10_m2 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise10_s1.npy'),(10))
no_noise_m2_s2 =  np.reshape(np.load('results/MAPEs_solution/m2/m2_no_noise_s2.npy'), (10,))

mean_no_noise_m2 = np.mean(no_noise_m2)
mean_noise1_m2 = np.mean(noise_1_m2)
mean_noise5_m2 = np.mean(noise_5_m2)
mean_noise10_m2 = np.mean(noise_10_m2)
mean_no_noise_m2_s2 = np.mean(no_noise_m2_s2)

all_noise_results_m2 = [no_noise_m2, noise_1_m2, noise_5_m2, noise_10_m2]
labels_noise = ['no noise', '1% noise', '5% noise', '10% noise']
plt.figure(3)
plt.boxplot(all_noise_results_m2, vert=False, labels=labels_noise)
plt.xlabel('MAPE')
#plt.title('Boxplots with MAPEs for different levels of noise (method 2)')




