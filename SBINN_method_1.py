# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:08:39 2022

@author: 20182460
"""

"""
Created on Wed May 25 14:34:27 2022

@author: 20182460
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from data_generation import calc_solution
from tabulate import tabulate

#%%
#choose the dataset
datafile = 'data_sets/data_no_noise.npy'
fn_test_sol = 'results/MAPEs_solution/m1_test/TEST'
fn_train_sol = 'results/MAPEs_solution/m1_train/TEST'

strategy_blood1 = [0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
strategy_blood2 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]
strategy_blood3 = []
tp = 0
for i in range(52):
    strategy_blood3.append(tp)
    tp += 7
  
strategy_tissue1 = [0, 50]
strategy_tissue2 = [0]
strategy_tissue3 = [0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360]

strategy_blood = strategy_blood1
strategy_tissue = strategy_tissue1
 
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
np.random.seed(1234)
tf.random.set_seed(1234)

#load the data 
data = np.load(datafile, allow_pickle = True).item()
t_total= np.zeros([len(data), len(data[0][0])])
x_total = np.zeros([len(data), len(data[0][1])])
y_total = np.zeros([len(data), len(data[0][2])])
params = np.zeros([len(data),4])
initials =  np.zeros([len(data),3])
for i in range (len(data)): #loop over the simulations
    t_total[i] = data[i][0]
    x_total[i] = data[i][1]
    y_total[i] = data[i][2]
    params[i][0] = data[i][3]
    params[i][1] = data[i][4]
    params[i][2] = data[i][5]
    params[i][3] = data[i][6]
    initials[i][0]  = data[i][7]
    initials[i][1] = data[i][8]
   
t = t_total
x = x_total
y = y_total
  
#split data into training data en testdata 
t_train, t_test, x_train, x_test, y_train, y_test, params_train, params_test, initials_train, initials_test = train_test_split(t, x, y, params, initials, test_size = 0.2, train_size = 0.8, random_state = 10)

norm_t_train = (t_train - t_train.min())/(t_train.max() - t_train.min()) 
norm_x_train = (x_train - x_train.min())/(x_train.max() - x_train.min()) 
norm_y_train = (y_train - y_train.min())/(y_train.max() - y_train.min()) 

norm_params_train = np.zeros_like(params_train)
norm_params_train[:,0] = (params_train[:,0] - params_train[:,0].min())/(params_train[:,0].max() - params_train[:,0].min())
norm_params_train[:,1] = (params_train[:,1] - params_train[:,1].min())/(params_train[:,1].max() - params_train[:,1].min()) 
norm_params_train[:,2] = (params_train[:,2] - params_train[:,2].min())/(params_train[:,2].max() - params_train[:,2].min()) 
norm_params_train[:,3] = (params_train[:,3] - params_train[:,3].min())/(params_train[:,3].max() - params_train[:,3].min()) 

norm_x_test = (x_test - x_train.min())/(x_train.max() - x_train.min()) 
norm_y_test = (y_test - y_train.min())/(y_train.max() - y_train.min()) 

#%%

# build model/network architecture
model = Sequential([
    Dense(80, activation = 'relu'),
    Dense(80, activation = 'relu'),
    Dense(80, activation = 'relu'),
    Dense(80, activation = 'relu'),
    Dense(80, activation = 'relu'), 
    Dense(80, activation = 'relu'), 
    Dense(80, activation = 'relu'), 
    Dense(80, activation = 'relu'), 
    Dense(4)
])

#loss function based on data (true parameter values vs predicted parameter values) 
def params_loss(params_true, params_pred):
    params_true_tot = np.concatenate((params_true, params_true))
    return tf.reduce_mean(tf.square(params_pred - params_true_tot))

#loss function based on system of ODEs
def PDE_loss (t, x, y, params_pred, model):

    # d1_pred = params_pred[:,0]
    # d2_pred = params_pred[:,1]
    # m_pred = params_pred[:,2]
    # c_pred = params_pred[:,3]
    #get individual predicted parameters 
    d1_pred = (params_pred[:,0])*(params_train[:,0].max()-params_train[:,0].min())+params_train[:,0].min()
    d2_pred = (params_pred[:,1])*(params_train[:,1].max()-params_train[:,1].min())+params_train[:,1].min()
    m_pred = (params_pred[:,2])*(params_train[:,2].max()-params_train[:,2].min())+params_train[:,2].min()
    c_pred = (params_pred[:,3])*(params_train[:,3].max()-params_train[:,3].min())+params_train[:,3].min()
    
    t = t*(t_train.max() - t_train.min())+t_train.min()
    x = x*(x_train.max() - x_train.min())+x_train.min()
    y = y*(y_train.max() - y_train.min())+y_train.min()
    
    t = np.concatenate((t, t))
    x = np.concatenate((x, x))
    y = np.concatenate((y, y))
    
    #calculate first derivative of x and y with respect to time
    x_t_train = np.empty(shape=(x.shape),dtype='float64')
    y_t_train = np.empty(shape=(y.shape),dtype='float64')
    for i in range(x.shape[0]):
        x_t_train[i] = np.gradient(x[i,:],t[i,:])
        y_t_train[i] = np.gradient(y[i,:],t[i,:])
    
    #Calculate residual
    f_x = tf.transpose(x_t_train) + tf.transpose(x)*m_pred + d1_pred*(tf.transpose(x)-c_pred)
    loss_x = tf.reduce_mean(tf.square(f_x - tf.zeros_like(f_x)))
    
    f_y  = tf.transpose(y_t_train) -  tf.transpose(x)*m_pred + d2_pred*tf.transpose(y)
    loss_y = tf.reduce_mean(tf.square(f_y - tf.zeros_like(f_y)))
    
    return loss_x, loss_y

def calc_total_loss(t, x, y, params_true, params_pred, model):
    
    loss_params =  params_loss(params_true, params_pred)
    loss_ODE1, loss_ODE2 = PDE_loss(t, x, y, params_pred, model) 
    loss_ODEs = 1e-21*(loss_ODE1 + loss_ODE2) 
    loss_total = loss_params + loss_ODEs
 
    return loss_total, loss_params, loss_ODEs

optimizer = Adam()

loss_history_total = []
loss_history_params = []
loss_history_ODEs = []
d1_history = []
d2_history = []
m_history = []
c_history = []

def train_step(t, x, y, params_true, model): #(y_true, p,t,q,q_t,p_t, model)
    with tf.GradientTape() as tape:
     
        y_input = np.zeros([len(y), len(strategy_blood)])
        x_input = np.zeros([len(x), len(strategy_blood)]) #give x the same shape as y to match. 
        for i in range(len(y)):
            for j in range(len(strategy_blood1)):
                timepoint = strategy_blood[j]
                blood_count = y[i,timepoint]
                y_input[i,j] = blood_count
            for k in range(len(strategy_tissue)):
                timepoint = strategy_tissue[k]
                blood_count = x[i,timepoint]
                x_input[i,k] = blood_count
        total_input = np.concatenate((x_input, y_input))
        
        params_pred = model(total_input)
        loss_total, loss_params, loss_ODEs = calc_total_loss(t, x, y, params_true, params_pred, model)
      
    loss_history_total.append(loss_total.numpy())
    loss_history_params.append(loss_params.numpy())
    loss_history_ODEs.append(loss_ODEs.numpy())
    
    d1_history.append(params_pred[:,0].numpy())
    d2_history.append(params_pred[:,1].numpy())
    m_history.append(params_pred[:,2].numpy())
    c_history.append(params_pred[:,2].numpy())
    
    grads = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train (epochs):
    for epoch in tqdm(range(epochs)):
        train_step(norm_t_train, norm_x_train, norm_y_train, norm_params_train, model)
        
train(1500)

#%% Evaluation TEST SET

# Plot loss
plt.figure()
#plt.yscale(('log'))
#plt.plot(loss_history_total, 'b', label='total loss')
plt.plot(loss_history_params, label='network loss')
plt.plot(loss_history_ODEs, label='residual loss')
plt.ylim(0,0.5)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('loss[-]')

#sample the time points from the test set
y_test_input = np.zeros([len(norm_y_test), len(strategy_blood)])
x_test_input = np.zeros([len(norm_x_test), len(strategy_blood)])
for i in range(len(norm_y_test)):
    for j in range(len(strategy_blood)):
        timepoint = strategy_blood[j]
        blood_count = norm_y_test[i,timepoint]
        y_test_input[i,j] = blood_count
    for k in range(len(strategy_tissue)):
        timepoint = strategy_tissue[k]
        blood_count = norm_x_test[i,timepoint]
        x_test_input[i,k] = blood_count
total_test_input = np.concatenate((x_test_input, y_test_input))

par_pred_test_norm = model.predict(total_test_input)

#scale parameters back 
params_pred_test = np.zeros_like(par_pred_test_norm)
params_pred_test[:,0] = (par_pred_test_norm[:,0])*(params_train[:,0].max()-params_train[:,0].min())+params_train[:,0].min()
params_pred_test[:,1] = (par_pred_test_norm[:,1])*(params_train[:,1].max()-params_train[:,1].min())+params_train[:,1].min()
params_pred_test[:,2] = (par_pred_test_norm[:,2])*(params_train[:,2].max()-params_train[:,2].min())+params_train[:,2].min()
params_pred_test[:,3] = (par_pred_test_norm[:,3])*(params_train[:,3].max()-params_train[:,3].min())+params_train[:,3].min()

#compare true parameters of test set with predicted parameters of test set 
list_rel_errors = [[],[],[],[]]

for i in range(len(params_test)): #200
    for j in range(len(params_test[0])): #4
        param_true = params_test[i][j]
        param_pred = params_pred_test[i][j]
        param_error = (abs((param_true-param_pred)/param_true))*100
        list_rel_errors[j].append(param_error)

list_mean_errors = []
list_std_errors = []
for i in range(len(list_rel_errors)):
    list_mean_errors.append(np.mean(list_rel_errors[i]))
    list_std_errors.append(np.std(list_rel_errors[i]))

table = {}
table['param'] = ['d1', 'd2', 'm', 'c']
table['mean abs rel error'] = list_mean_errors
table['std abs rel error'] = list_std_errors

print(tabulate(table, headers = 'keys'))

MAPEs = []
plot_sim_nr = 0
for i in range(len(initials_test)):
        
    x0 = initials_test[i][0]
    y0 = initials_test[i][1]
    Btot = initials_test[i][2]

    d1_pred = params_pred_test[i][0]
    d2_pred = params_pred_test[i][1]
    m_pred = params_pred_test[i][2]
    c_pred = params_pred_test[i][3]

    d1_true = params_test[i][0]
    d2_true = params_test[i][1]
    m_true = params_test[i][2]
    c_true = params_test[i][3]

    #calculate the predicted solution
    t_pred, x_pred, y_pred = calc_solution(x0, y0, d1_pred, d2_pred, m_pred, c_pred)
    t_true, x_true, y_true = calc_solution(x0, y0, d1_true, d2_true, m_true, c_true)
    
    #calculate the sum of squared errors (SSE) for all 500 datapoints
    errors = []
    for j in range(len(x_pred)): #loop over number of data points (500)
        x_predj = x_pred[j]
        x_truej = x_true[j]
        errorx = abs((x_truej-x_predj)/x_truej)*100
        errors.append(errorx)
        y_predj = y_pred[j]
        y_truej = y_true[j]
        errory = abs((y_truej-y_predj)/y_truej)*100
        errors.append(errory)
    
    N = len(x_pred) + len(y_pred)
    MAPE = (sum(errors))/N
    MAPEs.append(MAPE)
    
    if i == plot_sim_nr:
        plt.figure(2, figsize=(10,4))
        
        plt.subplot(1,2,1)
        plt.plot(t_pred, x_pred, 'b', label='predicted solution')
        plt.plot(t_true, x_true, 'g', label='true solution')
        #plt.text(100,3e12, 'MAPE = %f' % MAPE)
        plt.legend()
        plt.xlabel('days after treatment')
        plt.ylabel('total lymphocyte count in tissue')

        plt.subplot(1,2,2)
        plt.plot(t_pred, y_pred, 'b', label='predicted solution')
        plt.plot(t_true, y_true, 'g', label='true solution')
        #plt.text(200,6e11, 'MAPE = %f' % MAPE)
        plt.legend()
        plt.xlabel('days after treatment')
        plt.ylabel('total lymphocyte count in blood')

        plt.subplots_adjust(wspace=0.2)

        plt.show()
        
print(np.mean(MAPEs))
np.save(fn_test_sol, MAPEs)

#%% evaluation TRAINING SET

y_train_input = np.zeros([len(norm_y_train), len(strategy_blood)])
x_train_input = np.zeros([len(norm_x_train), len(strategy_blood)])
for i in range(len(norm_y_test)):
    for j in range(len(strategy_blood)):
        timepoint = strategy_blood[j]
        blood_count = norm_y_train[i,timepoint]
        y_train_input[i,j] = blood_count
    for k in range(len(strategy_tissue)):
        timepoint = strategy_tissue[k]
        blood_count = norm_x_train[i,timepoint]
        x_train_input[i,k] = blood_count
total_train_input = np.concatenate((x_train_input, y_train_input))

par_pred_train_norm = model.predict(total_train_input)

#scale parameters back 
params_pred_train = np.zeros_like(par_pred_train_norm)
params_pred_train[:,0] = (par_pred_train_norm[:,0])*(params_train[:,0].max()-params_train[:,0].min())+params_train[:,0].min()
params_pred_train[:,1] = (par_pred_train_norm[:,1])*(params_train[:,1].max()-params_train[:,1].min())+params_train[:,1].min()
params_pred_train[:,2] = (par_pred_train_norm[:,2])*(params_train[:,2].max()-params_train[:,2].min())+params_train[:,2].min()
params_pred_train[:,3] = (par_pred_train_norm[:,3])*(params_train[:,3].max()-params_train[:,3].min())+params_train[:,3].min()

#compare true parameters of test set with predicted parameters of test set 
list_rel_errors_train = [[],[],[],[]]

for i in range(len(params_train)): #200
    for j in range(len(params_train[0])): #4
        param_true = params_train[i][j]
        param_pred = params_pred_train[i][j]
        param_error = (abs((param_true-param_pred)/param_true))*100
        list_rel_errors_train[j].append(param_error)

list_mean_errors_train = []
list_std_errors_train = []
for i in range(len(list_rel_errors)):
    list_mean_errors_train.append(np.mean(list_rel_errors_train[i]))
    list_std_errors_train.append(np.std(list_rel_errors_train[i]))

table_train = {}
table_train['param'] = ['d1', 'd2', 'm', 'c']
table_train['mean abs rel error'] = list_mean_errors_train
table_train['std abs rel error'] = list_std_errors_train

print(tabulate(table_train, headers = 'keys'))

MAPEs_train = []
plot_sim_nr = 0
for i in range(len(initials_test)):
        
    x0 = initials_train[i][0]
    y0 = initials_train[i][1]
    Btot = initials_train[i][2]

    d1_pred = params_pred_train[i][0]
    d2_pred = params_pred_train[i][1]
    m_pred = params_pred_train[i][2]
    c_pred = params_pred_train[i][3]

    d1_true = params_train[i][0]
    d2_true = params_train[i][1]
    m_true = params_train[i][2]
    c_true = params_train[i][3]

    #calculate the predicted solution
    t_pred, x_pred, y_pred = calc_solution(x0, y0, d1_pred, d2_pred, m_pred, c_pred)
    t_true, x_true, y_true = calc_solution(x0, y0, d1_true, d2_true, m_true, c_true)
    
    #calculate the sum of squared errors (SSE) for all 500 datapoints
    errors = []
    for j in range(len(x_pred)): #loop over number of data points (500)
        x_predj = x_pred[j]
        x_truej = x_true[j]
        errorx = abs((x_truej-x_predj)/x_truej)*100
        errors.append(errorx)
        y_predj = y_pred[j]
        y_truej = y_true[j]
        errory = abs((y_truej-y_predj)/y_truej)*100
        errors.append(errory)
    
    N = len(x_pred) + len(y_pred)
    MAPE = (sum(errors))/N
    MAPEs_train.append(MAPE)
        
print(np.mean(MAPEs_train))
np.save(fn_train_sol, MAPEs_train)