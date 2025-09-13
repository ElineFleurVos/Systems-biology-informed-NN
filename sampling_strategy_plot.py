# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:54:41 2022

@author: 20182460
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

strategy_blood1 = [0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
strategy_blood2 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
strategy_blood3 = []
tp = 0
for i in range(52):
    strategy_blood3.append(tp)
    tp += 7
    
strategy_tissue1 = [0]
strategy_tissue2 = [0, 50]

blood_strategies = [strategy_blood1, strategy_blood2, strategy_blood3]
tissue_strategies = [strategy_tissue1, strategy_tissue2]

data = np.load('data_no_noise.npy', allow_pickle = True).item()

sim_nr = 15

data_t = data[sim_nr][0]
data_x = data[sim_nr][1]
data_y = data[sim_nr][2]

fig = plt.figure(1, figsize = (12,5))
#fig.suptitle("Example analytical solution with sampling strategies", size=16)

gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])

ax1 = fig.add_subplot(gs[0])
ax1.plot(data_t, data_x)
ax1.set_ylabel('x (total lymphocyte count in tissue)', size= 14)
ax1.set_xlim(-10,510)
ax1.tick_params(right = False, bottom = False, labelbottom = False)

ax2 = fig.add_subplot(gs[1])
ax2.plot(data_t, data_y)
ax2.set_ylabel('y (total lymphocyte count in blood)', size = 14)
ax2.set_xlim(-10,510)
ax2.tick_params(right = False, bottom = False, labelbottom = False)

ax3 = fig.add_subplot(gs[2])
j = 1
for i in range(len(tissue_strategies)):
    #color = colors[i]
    tissue_strategy = tissue_strategies[i]
    ax3.plot(tissue_strategies[i],[j]*len(tissue_strategy), '^')
    j += 1
ax3.set_xlim(-10,510)
ax3.set_ylim(0.5,2.5)
ax3.set_xlabel('days after treatment', size=14)
ax3.tick_params(labelleft = False, left = False, right = False)

ax4 = fig.add_subplot(gs[3])
k = 1
for i in range(len(blood_strategies)):
    #color = colors[i]
    blood_strategy = blood_strategies[i]
    ax4.plot(blood_strategies[i],[k]*len(blood_strategy), '.')
    k += 1
ax4.set_xlim(-10,510)
ax4.set_ylim(0,4)
ax4.set_xlabel('days after treatment', size=14)
ax4.tick_params(labelleft = False, left = False, right = False)

plt.subplots_adjust(wspace=0.2)
plt.subplots_adjust(hspace=0.005)