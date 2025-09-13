# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:41:56 2022

@author: 20182460
"""

from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy.stats import ranksums
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% shapiro wilk test
#H0: data was drawn from a normal distribution
#if p<0.05 reject null hypotheses

#make lists with the filenames 
results_filenames_m1 = os.listdir('results/MAPEs_solution/m1_test/') 
p_values_m1 = []
test_statistics_m1 = []
for i in range(len(results_filenames_m1)):
    filename = results_filenames_m1[i]
    data = np.load('results/MAPEs_solution/m1_test/' + filename)
    stat, p = shapiro(data)
    test_statistics_m1.append(stat)
    p_values_m1.append(p)

dict_pvalues_m1 = {'name': results_filenames_m1, 'test statistic': test_statistics_m1,'p-value': p_values_m1}  
df_pvalues_m1 = pd.DataFrame(dict_pvalues_m1)
    
results_filenames_m2 = os.listdir('results/MAPEs_solution/m2/') 
p_values_m2 = []
test_statistics_m2 = []
for i in range(len(results_filenames_m2)):
    filename = results_filenames_m2[i]
    data = np.load('results/MAPEs_solution/m2/' + filename)
    stat, p = shapiro(data)
    test_statistics_m2.append(stat)
    p_values_m2.append(p)

dict_pvalues_m2 = {'name': results_filenames_m2, 'test statistic': test_statistics_m2,'p-value': p_values_m2}  
df_pvalues_m2 = pd.DataFrame(dict_pvalues_m2)
    

#%% wilcoxon_signed_rank test 
#H0: two related paired samples come from the same distribution.
#if p<0.05 reject null hypothesis
         
def wilcoxon_signed_rank(data1, data2):
    '''
    A wilcoxon signed rank test can be performed to compare the results from two models
    
    The input should be two the results from the two models 
    
    The output is the P-value
    '''
    stat, p = wilcoxon(data1, data2)
    return stat, p

#results for noise test m1
m1_no_noise = np.load('results/MAPEs_solution/m1_test/m1_no_noise_b1_t1.npy') 
m1_noise1 = np.load('results/MAPEs_solution/m1_test/m1_noise1_b1_t1.npy') 
m1_noise5 = np.load('results/MAPEs_solution/m1_test/m1_noise5_b1_t1.npy') 
m1_noise10 = np.load('results/MAPEs_solution/m1_test/m1_noise10_b1_t1.npy') 

results_noise_m1 = [m1_no_noise, m1_noise1, m1_noise5, m1_noise10]

stat_, p_ = wilcoxon_signed_rank(m1_noise5, m1_noise10)
print(stat_)
print(p_)

def heatmap_noise(results_noise):
    p_wilcoxon_no_noise = []
    p_wilcoxon_noise1 = []
    p_wilcoxon_noise5 = []
    #p_wilcoxon_m1_noise10 = []
    for i in range(len(results_noise)):
        result1 = results_noise[i]
        for j in range(i+1, len(results_noise)):
            result2 = results_noise[j]
    
            try:
                stat, p = wilcoxon_signed_rank(result1, result2)
                #stat_wilcoxon_m1.append(stat)
            except ValueError:
                p = 0,0
                
            # if p>0.01:
            #     p = round(p,3)
            # elif p>= 0.001:
            #     p = round(p,2)
            # else:
            #     p = 0.001
                
            #print(p)
            if i == 0:
                p_wilcoxon_no_noise.append(p)
            if i == 1:
                p_wilcoxon_noise1.append(p)
            if i == 2:
                p_wilcoxon_noise5.append(p)
            #if j == 3:
                #p_wilcoxon_m1_noise10.append(p)
                    
    p_wilcoxon_noise5.insert(0,0)
    p_wilcoxon_noise5.insert(0,0)
    p_wilcoxon_noise1.insert(0,0)
    dict_wilcoxon_m1_noise = {'no noise': p_wilcoxon_no_noise, '1%': p_wilcoxon_noise1,
                              '5%': p_wilcoxon_noise5}#, '10%': p_wilcoxon_m1_noise10}  
    df_wilcoxon_m1_noise = pd.DataFrame(dict_wilcoxon_m1_noise)
    y_axis_labels =['1%', '5%', '10%']
    
    mask = np.zeros_like(df_wilcoxon_m1_noise)
    mask[(0,1)] = True
    mask[(0,2)] = True
    mask[(1,2)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(3,4),)
        ax = sns.heatmap(df_wilcoxon_m1_noise, mask=mask, square=True, annot=True,
                          yticklabels = y_axis_labels, cbar=True,cbar_kws={"shrink": .65})

heatmap_noise(results_noise_m1)

#results for noise test m2
m2_no_noise = np.reshape(np.load('results/MAPEs_solution/m2/m2_no_noise_s1.npy'), (10,))
m2_noise1 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise1_s1.npy'), (10,))
m2_noise5 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise5_s1.npy'), (10,))
m2_noise10 = np.reshape(np.load('results/MAPEs_solution/m2/m2_noise10_s1.npy'), (10,))

results_noise_m2 = [m2_no_noise, m2_noise1, m2_noise5, m2_noise10] 

heatmap_noise(results_noise_m2)

#sampling results m1

folder_test = 'results/MAPEs_solution/m1_test/'
strategy_11 = np.load(folder_test + 'm1_no_noise_b1_t1.npy')
strategy_21 = np.load(folder_test + 'm1_no_noise_b2_t1.npy')
strategy_31 = np.load(folder_test + 'm1_no_noise_b3_t1.npy')
strategy_12 = np.load(folder_test + 'm1_no_noise_b1_t2.npy')
strategy_22 = np.load(folder_test + 'm1_no_noise_b2_t2.npy')
strategy_32 = np.load(folder_test + 'm1_no_noise_b3_t2.npy')
strategy_13 = np.load(folder_test + 'm1_no_noise_b1_t3.npy')

results_sampling_m1 = [strategy_11, strategy_21, strategy_31, strategy_12, strategy_22, 
                       strategy_32, strategy_13]

def heatmap_sampling(results_sampling):
    p_wilcoxon_11 = []
    p_wilcoxon_21 = []
    p_wilcoxon_31 = []
    p_wilcoxon_12 = []
    p_wilcoxon_22= []
    p_wilcoxon_32 = [] 
    for i in range(len(results_sampling)):
        result1 = results_sampling[i]
        for j in range(i+1, len(results_sampling)):
            result2 = results_sampling[j]
    
            try:
                stat, p = wilcoxon_signed_rank(result1, result2)
                #stat_wilcoxon_m1.append(stat)
            except ValueError:
                p = 0,0
            
            if p <0.01:
                p = round(p,3)
                
            #print(p)
            if i == 0:
                p_wilcoxon_11.append(p)
            if i == 1:
                p_wilcoxon_21.append(p)
            if i == 2:
                p_wilcoxon_31.append(p)
            if i == 3:
                p_wilcoxon_12.append(p)
            if i == 4:
                p_wilcoxon_22.append(p)
            if i == 5:
                p_wilcoxon_32.append(p)
            
            
    for i in range(5):             
        p_wilcoxon_32.insert(0,0)
    
    for i in range(4):
        p_wilcoxon_22.insert(0,0)
    
    for i in range(3): 
        p_wilcoxon_12.insert(0,0)
    
    for i in range(2):
        p_wilcoxon_31.insert(0,0)
        
    p_wilcoxon_21.insert(0,0)
    
    dict_wilcoxon_m1_sampling = {'11': p_wilcoxon_11, '21': p_wilcoxon_21,
                              '31': p_wilcoxon_31, '12': p_wilcoxon_12,
                              '22': p_wilcoxon_22, '32': p_wilcoxon_32}  
    df_wilcoxon_m1_sampling = pd.DataFrame(dict_wilcoxon_m1_sampling)
    y_axis_labels =['21', '31', '12', '22', '32', '13']
    
    mask = np.zeros_like(df_wilcoxon_m1_sampling)
    mask[(0,1)]= True
    mask[(0,2)] = True
    mask[(0,3)] = True
    mask[(0,4)] = True
    mask[(0,5)] = True
    mask[(1,2)] = True
    mask[(1,3)] = True
    mask[(1,4)] = True
    mask[(1,5)] = True
    mask[(2,3)] = True
    mask[(2,4)] = True
    mask[(2,5)] = True
    mask[(3,4)] = True
    mask[(3,5)] = True
    mask[(4,5)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,4),)
        ax = sns.heatmap(df_wilcoxon_m1_sampling, mask=mask, square=True, annot=True,
                          yticklabels = y_axis_labels, cbar=True,cbar_kws={"shrink": .8})

heatmap_sampling(results_sampling_m1)

#%% Wilcoxon rank sum test
#H0: two independent samples come from the same distribution.
#if p<0.05 reject null hypothesis

stat, p = ranksums(m2_no_noise, strategy_13)
print(stat)
print(p)

