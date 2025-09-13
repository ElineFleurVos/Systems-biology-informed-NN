# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:27:09 2022

@author: 20182460
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
from analytical_solution import calc_solution
from tabulate import tabulate

#%%

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, Time, uy, ux, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        # self.x = X[:,0:1]
        self.t = Time
        self.uy = uy
        self.ux = ux
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False))
        
        # Initialize parameters
        self.d1 = tf.Variable([2.7], dtype=tf.float32) #2.7e-2
        self.d2 = tf.Variable([2.7], dtype=tf.float32) #1.7e-2
        self.m = tf.Variable([8], dtype = tf.float32) #8e-3
        self.c = tf.Variable([1.698], dtype = tf.float32) #1.698e11
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.uy_tf = tf.placeholder(tf.float32, shape=[None, self.uy.shape[1]])
        self.ux_tf = tf.placeholder(tf.float32, shape=[None, self.ux.shape[1]])
    
        self.uy_pred, self.uy_t_pred, self.ux_pred, self.ux_t_pred = self.net_u(self.t_tf)
        self.f1_pred, self.f2_pred = self.net_f(self.t_tf)
        
        # loss output nn
        self.loss_n1 = tf.reduce_mean(tf.square(self.uy_tf - self.uy_pred))
        self.loss_n2 = tf.reduce_mean(tf.square(self.ux_tf  - self.ux_pred))
        self.loss_n = self.loss_n1 + self.loss_n2
        
        # loss BC
        self.loss_f1 = tf.reduce_mean(tf.square(self.f1_pred))
        self.loss_f2 = tf.reduce_mean(tf.square(self.f2_pred))
        self.loss_f = self.loss_f1 + self.loss_f2
        
        #total loss 
        self.loss = self.loss_n + self.loss_f 
    
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
       
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y
            
    def net_u(self, t):# x, t):  
        u = self.neural_net(t, self.weights, self.biases)
        ux = u[:,0:1]
        uy = u[:,1:2]
        uy_t = tf.gradients(uy,t)
        ux_t = tf.gradients(ux,t)
        
        return uy, uy_t, ux, ux_t
    
    
    def net_f(self,  t):
        d1 = self.d1 #*1e-2   
        d2 = self.d2 #*1e-2
        m = self.m #*1-3
        c = self.c #*1e11
      
        uy, uy_t, ux, ux_t = self.net_u(t)
        
        f1 = ux_t + m*ux + d1*(ux-c)
        f2 = uy_t - m*ux + d2*uy
        
        return f1, f2
        
    def callback(self,  loss_n, loss_f, d1, d2, m, c):
        print(' Loss_n: %e, Loss_f: %e, d1: %.5f, d2: %.5f, m: %5f, c: %5f' % (loss_n, loss_f, d1, d2, m, c))
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.ux_tf:self.ux,  self.uy_tf: self.uy}#{self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
        
        start_time = time.time()
        d1_values = np.zeros(nIter)
        d2_values = np.zeros(nIter)
        m_values = np.zeros(nIter)
        c_values = np.zeros(nIter)
        loss_n_values = np.zeros(nIter)
        loss_f_values = np.zeros(nIter)
        #loss_values = np.zeros(nIter) 
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                #loss_value = self.sess.run(self.loss, tf_dict)
                loss_n_value = self.sess.run(self.loss_n, tf_dict)
                loss_f_value = self.sess.run(self.loss_f, tf_dict)
                d1_value = self.sess.run(self.d1)
                d2_value = self.sess.run(self.d2)
                m_value = self.sess.run(self.m)
                c_value = self.sess.run(self.c)
                
                print('It: %d, Loss_n: %.3e, Loss_f: %.3e, d1: %.3f, d2: %.6f, m: %.3f, c: %3f, Time: %.2f' % 
                      (it, loss_n_value, loss_f_value, d1_value, d2_value, m_value, c_value, elapsed))
                start_time = time.time()
                
                d1_values[it] = d1_value
                d2_values[it]= d2_value
                m_values[it] = m_value
                c_values[it] = c_value
                #loss_values[it] = loss_value
                loss_n_values[it] = loss_n_value
                loss_f_values[it] = loss_f_value
        
        return d1_values, d2_values, m_values, c_values, loss_n_values, loss_f_values
           
    def predict(self, Time):
        
        tf_dict = {self.t_tf: Time}
        
        ux_p = self.sess.run(self.ux_pred, tf_dict)
        uy_p = self.sess.run(self.uy_pred, tf_dict)
    
        return ux_p, uy_p

Data = np.load('data_sets/data_no_noise.npy', allow_pickle = True).item()
file_save_MAPE = 'results/MAPEs_solution/m2/TEST'

simulation_numbers = [2,387,245,360,410,58,196,200,331,67]

nr_iterations = 20000
sampling1 = np.array([0, 14, 28, 42, 56, 70, 84, 98, 130, 160, 190, 220, 250, 270, 300, 330, 360])
sampling2 = np.array([0, 28, 50, 100, 200, 300])
sampling = sampling2
   
MAPEs = []
errors_d1 = []
errors_d2 = []
errors_m = []
errors_c = []

plot_sim_nr = 0
for i in range(len(simulation_numbers)):
    sim_nr = simulation_numbers[i]
    print(sim_nr)
    layers = [1, 80, 80, 80, 80, 80, 80, 80, 80, 2]

    t = Data[sim_nr][0]
    x = Data[sim_nr][1] *1e-12
    y = Data[sim_nr][2] *1e-11
    
    T = t
    t = T[:,None]
    
    x_star = x.flatten()[:,None] 
    y_star = y.flatten()[:,None]        
    
    #Domain bounds
    lb = t.min(0)
    ub = t.max(0)             

    t_train = t[sampling]
    y_train = y_star[sampling]
    x_train= x_star[sampling]
    
    model = PhysicsInformedNN(t_train, y_train, x_train, layers, lb, ub)
    results = model.train(nr_iterations)
    
    ux_pred, uy_pred = model.predict(t)      
    
    loss_n_values = results[4]
    loss_r_values = results[5]
    
    loss_n_values = [k for k in loss_n_values if k!=0]
    loss_r_values = [k for k in loss_r_values if k!=0]
    
    d1_true = Data[sim_nr][3]
    print(d1_true)
    d2_true = Data[sim_nr][4]
    m_true = Data[sim_nr][5]
    c_true = Data[sim_nr][6]
    x0 = Data[sim_nr][7]
    print(x0)
    y0 = Data[sim_nr][8]
       
    x_pred = ux_pred*1e12
    y_pred = uy_pred*1e11
    t_true, x_true, y_true = calc_solution(x0, y0, d1_true, d2_true, m_true, c_true)
    # x_true = Data[sim_nr][1]
    # y_true = Data[sim_nr][2]
    
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
    print(MAPE)
    MAPEs.append(MAPE)
    
    d1_pred = model.sess.run(model.d1)*1e-2
    d2_pred = model.sess.run(model.d2)*1e-2
    m_pred = model.sess.run(model.m)*1e-3
    c_pred = model.sess.run(model.c)*1e11
    
    errors_d1.append(abs((d1_pred - d1_true)/d1_true)*100)
    errors_d2.append(abs((d2_pred-d2_true)/d2_true)*100)
    errors_m.append(abs((m_pred-m_true)/m_true)*100)
    errors_c.append(abs((c_pred-c_true)/c_true)*100)
    
    #errors = [error_d1, error_d2, error_m, error_c]
    # table = {}
    # table['param'] = ['d1', 'd2', 'm', 'c']
    # table['abs rel error'] = errors
    # print(tabulate(table, headers = 'keys'))
    
    if i == plot_sim_nr:
        
        plt.figure(1)
        plt.yscale(('log'))
        plt.plot(loss_n_values, label='network loss')
        plt.plot(loss_r_values, label='residual loss')
        plt.legend()
        plt.xlabel('Iterations (x100)')
        plt.ylabel('loss[-]')
        
        plt.figure(2, figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(t, x_pred, 'b', label='predicted solution')
        plt.plot(t, x_true, 'g', label='true solution')
        plt.legend()
        plt.xlabel('days after treatment')
        plt.ylabel('total lymphocyte count in tissue')

        plt.subplot(1,2,2)
        plt.plot(t, y_pred, 'b', label='predicted solution')
        plt.plot(t, y_true, 'g', label='true solution')
        plt.legend()
        plt.xlabel('days after treatment')
        plt.ylabel('total lymphocyte count in blood')

        plt.subplots_adjust(wspace=0.2)

        plt.show()
        
        # ttrue, xtrue, ytrue = calc_solution(x0, y0, d1_true, d2_true, m_true, c_true)
        # tpred, xpred, ypred = calc_solution(x0, y0, d1_pred, d2_pred, m_pred, c_pred)
        
        # plt.figure()
        # plt.plot(ttrue, xtrue)
        # plt.plot(tpred, xpred)
        # plt.legend(['true solution', 'predicted solution'])
        # plt.xlabel('time [days]')
        # plt.ylabel('lymphocyte cell count in tissue')
        
        # plt.figure()
        # plt.plot(ttrue, ytrue)
        # plt.plot(tpred, ypred)
        # plt.legend(['true solution', 'predicted solution'])
        # plt.xlabel('time [days]')
  
        # plt.ylabel('lymphocyte cell count in blood')

#%%
np.save(file_save_MAPE, MAPEs)

list_mean_errors = []
list_std_errors = []
all_param_errors = [errors_d1, errors_d2, errors_m ,errors_c]
for i in all_param_errors:
    list_mean_errors.append(np.mean(i))
    list_std_errors.append(np.std(i))

table = {}
table['param'] = ['d1', 'd2', 'm', 'c']
table['mean MAPE'] = list_mean_errors
table['std MAPE'] = list_std_errors
print(tabulate(table, headers = 'keys'))