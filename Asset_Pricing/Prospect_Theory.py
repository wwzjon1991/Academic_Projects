# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:43:33 2017

@author: Jon Wee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as opt

Rf = 1.0303
delta = 0.99
gamma = 1
lamda = 2
rho = -np.log(delta)

## aggregate consumption growth
samples = 1000
epsilon = np.random.standard_normal(samples)

lng = 0.02+0.02*epsilon

## Market Portfolio

b0_size = 100
b0 = np.linspace(0,10,100)

# loss aversion function  
def v(R_t1):
    x_t1 = np.empty([])
    for i in range(len(R_t1)):
        if (R_t1[i]-Rf)>=0:
            x_t1 = np.append(x_t1 , R_t1[i]-Rf, axis=None)
            
        else:
            #x_t1[i] = lamda*R_t1[i]-Rf
            x_t1 = np.append(x_t1, lamda*(R_t1[i]-Rf), axis=None)
    x_t1 = np.delete(x_t1, 0, 0)   
    return x_t1

"""
######   Hardcoded    ##############################
def err(x,b):
    return 0.99*b*np.mean(v(x*np.exp(lng)))+0.99*x-1


# bisection function
def bisection(a, b, b00):
    
    if err(a,b00)*err(b,b00)<0:
        x_minus,x_plus = a,b
        middle = 0.5*(x_minus+x_plus)
        while np.abs(err(middle,b00)) > 1e-4:
            if (err(x_plus,b00)*err(middle,b00))<0:
                x_minus = middle
                middle = 0.5*(x_plus+x_minus)
            else:
                x_plus = middle
                middle = 0.5*(x_plus+x_minus)
        return middle                  
              
    else:
        return print('Error: f(a) and f(b) have the same sign')
                

# Solving bisection function   
opt_x = np.empty([100,0])
for i in b0:
    solve_x = bisection(1, 1.1, i)
    opt_x = np.append(opt_x, solve_x)

"""
# Using Scipy function
# Optimise X
opt_x = np.empty([100,0])
for i in range(len(b0)):
    solve_x = opt.bisect(lambda x: 0.99*b0[i]*np.mean(v(x*np.exp(lng)))+0.99*x-1, a= 1, b=1.1)
    opt_x= np.append(opt_x, solve_x)

    
# Price Dividend Ratio    
Price_Div = 1/(opt_x-1)        
PD_ratio = pd.DataFrame([Price_Div,b0], index=['PD Ratio', 'b0'])
PD_ratio = PD_ratio.transpose()


# Equity Premium Ratio
Ex_Mkt_Ret = np.empty([100,0])
for i in opt_x:
    Ex_Mkt_Ret = np.append(Ex_Mkt_Ret,np.mean(i*np.exp(lng)))
    Ex_Mkt_Ret1 = Ex_Mkt_Ret-Rf
Eqty_Prem = pd.DataFrame([Ex_Mkt_Ret1,b0], index=['Mkt Ret','b0'])
Eqty_Prem = Eqty_Prem.transpose()



# Plot GRaph
plt.figure(figsize=(12,8))
plt.plot(PD_ratio['b0'],PD_ratio['PD Ratio'],'b--')
plt.grid(False)
plt.xlabel('b0', fontsize = 20)
plt.ylabel('Price/Divdend', fontsize = 20)
plt.title('Price to Dividend ratio for the market portfolio', fontsize=20)
#plt.xlim(0.,10,20)
#plt.ylim(65,100,10)


plt.figure(figsize=(12,8))
plt.plot(Eqty_Prem['b0'],Eqty_Prem['Mkt Ret'],'b--')
plt.grid(False)
plt.xlabel('b0', fontsize = 20)
plt.ylabel('Equity premium', fontsize = 20)
plt.title('Equity Premium ratio for the market portfolio', fontsize=20)
#plt.xlim(0,10,20)
#plt.ylim(0, 0.007,10)


