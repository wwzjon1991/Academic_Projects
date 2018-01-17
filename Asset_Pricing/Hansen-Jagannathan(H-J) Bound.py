# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:47:43 2017

@author: Jon Wee
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from scipy import stats

## ln consumption growth
samples = 1000
#epsilon =  np.random.normal(loc=0, scale=1,samples)
# random function 
epsilon = np.random.standard_normal(samples)  # standard normal distribution
nu = np.random.choice([0,np.log(0.65)], p=[0.983,0.017],size=samples) # random with a particular probability

lng = 0.02+0.02*epsilon+nu

########################################################################################
# Hansen-Jagannathan Bound
gamma_size = 1000
gamma1 = np.round(np.linspace(1,4,gamma_size),3)


Ke_ratio = []

for i in range(gamma_size):
    M = 0.99*(np.exp(lng))**(-gamma1[i])
    SDM = np.std(M)
    MeanM = np.mean(M)
    Ke_ratio = np.append(Ke_ratio, SDM/MeanM)


idx = np.searchsorted(Ke_ratio, 0.4)
print("Hansen-Jagannathan bound:", Ke_ratio[idx],gamma1[idx])

#########################################################################################

gamma2 = np.linspace(1,7,gamma_size)

Price_Div = []
for i in range(gamma_size):
    xx = np.mean(0.99*(np.exp(lng))**(1-gamma2[i]))
    Price_Div = np.append(Price_Div, xx)

##########################################################################################

Equity_Prem = []

for i in range(gamma_size):
    M2 = (0.99*(np.exp(lng))**(-gamma2[i]))
    mkt_ret = (1/Price_Div[i])*(np.mean(np.exp(lng)))
    Rf = 1/(np.mean(M2))
    Equity_Prem = np.append(Equity_Prem, mkt_ret-Rf)


###############           GRAPH                   ##########################################
plt.figure(figsize=(12 , 8))
    
plt.plot(lng)
plt.xlabel('Events', fontsize = 20)
plt.ylabel('Consumption Growth', fontsize = 20)   
plt.title('Consumption Growth with rare disasters', fontsize=20) 
plt.xlim(-1,1000,20)
###############           PART 1                  ##########################################
plt.figure(figsize=(12 , 8))

plt.plot(gamma1,Ke_ratio, 'b')
plt.plot([0,5],[0.4,0.4],'r', linestyle = '--')
plt.plot(gamma1[idx],Ke_ratio[idx], 'ro', markersize = 10)
plt.text(gamma1[idx]-0.7,Ke_ratio[idx]+0.005, s = 'Min Gamma: %s' % np.round(gamma1[idx],3), fontsize=15 )

plt.grid(False)
plt.xlabel('Gamma', fontsize = 20)
plt.ylabel('SD(M)/E(M)', fontsize = 20)
plt.title('Hansen–Jagannathan Bound', fontsize=20)
#plt.xlim(1,4,20)
#plt.ylim(0,0.65,10)
#plt.legend()
plt.show()

##############             PART 2                 #############################################################
plt.figure(figsize=(12,8))

plt.plot(gamma2,Price_Div, 'b')    
plt.grid(False), 'b'
plt.xlabel('Gamma', fontsize = 20)
plt.ylabel('Price-Dividend Ratio', fontsize = 20)
plt.title('Price-Dividend Ratio', fontsize=20)
#plt.xlim(1,7,20)
#plt.ylim(0.9,1.1,10)
plt.show()        
         
##############            PART 3                 #############################################################

plt.figure(figsize=(12,8))

plt.plot(gamma2,Equity_Prem, 'b')    
plt.grid(False)
plt.xlabel('Gamma', fontsize = 20)
plt.ylabel('Equity Premium', fontsize = 20)
plt.title('Equity Premium', fontsize=20)
#plt.xlim(1,7,20)
#plt.ylim(0,0.15,10)
plt.show()

            
            