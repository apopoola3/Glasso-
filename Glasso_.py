#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:55:26 2021

@author: anjolaoluwapopoola
"""

import numpy as np
import sklearn as sklearn
import copy

# =============================================================================
# Example to generate psd covariance matrix first and then data 
# =============================================================================
np.random.seed(42)
precision = sklearn.datasets.make_sparse_spd_matrix(10, alpha=0.5,random_state=None)
cova = np.linalg.inv(precision) 

# =============================================================================
# Start by inputing data
# =============================================================================
np.random.seed(42)
error = np.random.normal(0,0.01,[200,10])
data = (np.random.multivariate_normal(mean=[0, 0, 0, 0,0,0,0,0,0,0], cov=cova, size=200)) + error
np.min(data)
data_num_rows = data.shape[0] 
data_num_features = data.shape[1] 
lambdaa = 0.1
n = np.arange(0,data_num_features)
p = 0.01

# =============================================================================
# Getting the covariance and precision matrix of data
# =============================================================================
S = np.cov(data,rowvar=False)

def softhres(z,gamma):
    
    if z > gamma:
        m = z - gamma
        
    elif z < -gamma:
        m = z + gamma
    
    else:
        m = 0
     
    return m

# =============================================================================
# Initializing
# =============================================================================
W = copy.deepcopy(S)
k = 0 
e = 1 
theta = np.linalg.inv(copy.deepcopy(S))
e_k = [1]



while e > p: #for w updates
    k = k + 1
    W_old = copy.deepcopy(W)
    
    for i in n:
    
        s_12 = S[i,:][n != i].reshape(len(n)-1,1)
        w_11_not = W[n!=i]
        w_11 = w_11_not[:,n!=i]
        w_22 = W[i,i]
        w_12 = W[i,:][n != i].reshape(len(n)-1,1)
        w_12t = w_12.transpose()
        q = np.arange(0,w_11.shape[1])
        np.random.seed(42)
        x = np.random.normal(0,1,(len(n)-1,1))
        y = 1
        k_2 = 0
        y_k = [1]
        
        while y > p: #for beta updates
            k_2 = k_2 + 1
            x_old = copy.deepcopy(x)
            
                
            for j in q:
                Aj = w_11[j,:].reshape((1,w_11.shape[0]))
                ata = w_11[j,j].reshape((1,1))
                gamma = lambdaa / ata
                az_0 = Aj[:,q!=j]
                az_2 = x[q!=j,:]
                az1 = np.dot(az_0,az_2)
                az3 = s_12[j]- az1                 
                z= az3/ata
                x[j] = softhres(z,gamma)
                
            x_new= copy.deepcopy(x)
            y = np.linalg.norm(x_new-x_old)
            y_k.append(y)
            
            
        w_12 = np.dot(w_11,x)
        
        print('w_12 = ',w_12)
        
        W[i,i] = w_22
        W[i,:][n!= i,np.newaxis]= w_12.reshape(len(n)-1,1)
        w_12t = w_12.transpose()
        W[:,i][np.newaxis,n!=i] = w_12t.reshape(1,len(n)-1)
        
        
        theta_22 = 1/(w_22 - np.matmul(w_12t,x))
        theta[i,i] = theta_22
        theta_12 =(-1)* np.matmul(x,theta_22) 
        theta[i,:][n!= i,np.newaxis]=theta_12.reshape(len(n)-1,1)
        theta_12t = theta_12.transpose()
        theta[:,i][np.newaxis,n!=i] = theta_12t.reshape(1,len(n)-1)
        
   
    W_new = copy.deepcopy(W)
    e = np.linalg.norm(W_new - W_old)
    e_k.append(e)
    
