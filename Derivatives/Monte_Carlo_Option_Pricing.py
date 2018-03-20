# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:40:43 2018

@author: Jon Wee
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

rf = 0.03
Time = 1
volatility = 0.25
S0 = 30
strike = 31
n = np.linspace(10,360,36,dtype=int)

#%%  Analytical Solution
# Black Scholes Call option
def BS_call( S, K, r, sigma, T) :
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_value

def BlackCallDelta(S, K, r, sigma, T, epsilon):
    return (BS_call(S+epsilon, K, r, sigma, T) - BS_call(S-epsilon, K, r, sigma, T)) / (2 * epsilon)

Analytical_call = BS_call(S0,strike,rf,volatility,Time)
#%% Question 1 Jarrow & Rudd Binomial Tree Method

def J_and_R(S, K, r, sigma, T, steps):
    dt = T/steps
    up = np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt))
    down = np.exp((r-0.5*sigma**2)*dt - sigma*np.sqrt(dt))
    P = 0.5
    x = np.log(S)
    x_u = np.log(up)
    x_d = np.log(down)

    St = []
    for i in range(0,steps+1):
        St = np.append(St, np.exp(x_u*(steps-i)+(x_d*(i))+x))
        

    Call = np.zeros((steps+1,steps+1))
    Call[steps,:]=np.maximum(0, St-K)
    
    
    DF = np.exp(-r*dt)
    for i in range(steps-1,0-1,-1):
        #print(i)                   up                  down
        Call[i,0:i+1] =  DF*(P*Call[i+1,:i+1]+P*Call[i+1,1:i+2])
            
    Call_value = Call[0][0]
    #return x_u, x_d, P, St , Call_value
    return Call_value
#JJ = J_and_R(S0, strike, rf, volatility, Time, 4 )
#%% Question 1 TRG
def TRG(S, K, r, sigma, T, steps):
    dt = T/steps
    nu = r-0.5*(sigma**2)
    x = np.log(S)
    x_delta = np.sqrt((sigma**2)*dt + (nu**2)*(dt**2))
    
    P_u = 0.5 + 0.5*(nu*dt/x_delta)
    P_d = 0.5 - 0.5*(nu*dt/x_delta)
    # stock price at maturity
    St = []
    #expectd stock prices at time T
    for i in range(0,steps+1):
        St = np.append(St, np.exp(x_delta*(steps-i)-(x_delta*(i))+x))
        
    Call = np.zeros((steps+1,steps+1))
    Call[steps,:]=np.maximum(0, St-K)
    
    
    DF = np.exp(-r*dt)
    for i in range(steps-1,0-1,-1):
        #print                          up           ddown          
        Call[i,0:i+1] =  DF*(P_u*Call[i+1,:i+1]+P_d*Call[i+1,1:i+2])
        
    Trg_value = Call[0,0]
    #return P_u, P_d, x_delta, St,Trg_value
    return Trg_value

#%% Question 2 Implicit method

def Implicit(S, K, r, sigma, T, steps):
    dt = T/steps
    nu = r - 0.5*(sigma**2)

    dx = sigma*np.sqrt(3*dt)
    
    N  = steps
    Nx = int(np.ceil(np.sqrt(3*N)))

    l_u = -0.5*( (sigma**2*dt/dx**2) + nu*dt/dx)
    l_d = -0.5*( (sigma**2*dt/dx**2) - nu*dt/dx)
    l_m = 1 + sigma**2*dt/(dx**2) + r*dt
    
    #Coefficient Matrix
    C_M = np.zeros((1+2*Nx,1+2*Nx))
    C_M[0, 0], C_M[0, 1] = 1,-1
    for row in range(1,len(C_M)-1): 
        C_M[row, row-1:row+2] = [l_u]+[l_m]+[l_d]
    C_M[-1, -2], C_M[-1, -1] = 1,-1
    inv_A = np.linalg.inv(C_M)
        
        

    St = np.zeros((1+2*Nx,1))
    St[-1,0] = S*np.exp(-dx*Nx) 
    for i in range(1+2*Nx-2, -1, -1):
        #print(i)
        St[i,0] = St[i+1,0]*np.exp(dx)
            
    # Call value at maturity  
    Call_M=np.maximum(0, St-K) 
            
    # Create matrix 
    Call = np.zeros((1+2*Nx,N+1))    
            
    Lambda_U = Call_M[0,0]-Call_M[1,0]
    Lambda_L = Call_M[-2,0]-Call_M[-1,0]
    for i in range(N-1,0-1,-1):
        Call_M[0,0] = Lambda_U
        Call_M[-1,0] = Lambda_L
        Call[:,i] = np.dot( inv_A,Call_M).T
        Call_M[:,0] = Call[:,i]

    sol = Call[Nx,0]
    return sol 

#ASD = Implicit(S0, strike, rf, volatility, Time, 1000 )

#%% Question 2 Crank method
def Crank(S, K, r, sigma, T, steps):
    dt = T/steps
    nu = r - 0.5*(sigma**2)
    
    dx = sigma*np.sqrt(3*dt)
    
    N  = steps
    Nx = int(np.ceil(np.sqrt(3*N)))
    
    q_u = -0.25*( (sigma**2*dt/dx**2) + nu*dt/dx)
    q_d = -0.25*( (sigma**2*dt/dx**2) - nu*dt/dx)
    q_m = 1 + 0.5*sigma**2*dt/(dx**2) + 0.5*r*dt
    
    #Coefficient Matrix
    C_M = np.zeros((1+2*Nx,1+2*Nx))
    C_M[0, 0], C_M[0, 1] = 1,-1
    for row in range(1,len(C_M)-1): 
        C_M[row, row-1:row+2] = [q_u]+[q_m]+[q_d]
    C_M[-1, -2], C_M[-1, -1] = 1,-1
    inv_A = np.linalg.inv(C_M)
        
        
    
    St = np.zeros((1+2*Nx,1))
    St[-1,0] = S*np.exp(-dx*Nx) 
    for i in range(1+2*Nx-2, -1, -1):
        #print(i)
        St[i,0] = St[i+1,0]*np.exp(dx)

    # Call value at maturity  
    Call_M=np.maximum(0, St-K) 
    
    # Create matrix 
    Call = np.zeros((1+2*Nx,N+1))    
 
    
    Lambda_U = Call_M[0,0]-Call_M[1,0]
    Lambda_L = Call_M[-2,0]-Call_M[-1,0]
    for i in range(N-1,-1,-1):
        Call_M[1:-1,0] = (-q_u*Call_M[0:1+2*Nx-2,0]-(q_m-2)*Call_M[1:1+2*Nx-1,0]-q_d*Call_M[2:1+2*Nx,0])
        Call_M[0,0] = Lambda_U
        Call_M[-1,0] = Lambda_L
        Call[:,i] = np.dot( inv_A,Call_M[:,0])
        Call_M[:,0] = Call[:,i]  
    sol = Call[Nx,0]
    return sol

#ASD = Crank(S0, strike, rf, volatility, Time, 10 )

#%%
Value = pd.DataFrame(index=range(0,36),columns=['Steps','JR_call',\
                     'TRG_call','Crank','Implicit','JR_err','TRG_err','Crank_err','Implicit_err'])
for id,i in enumerate(n):
    Value.loc[id]['Steps'] = i
    Value.loc[id]['JR_call']  = J_and_R(S0, strike, rf, volatility, Time, i )
    Value.loc[id]['TRG_call'] = TRG(S0, strike, rf, volatility, Time, i )
    Value.loc[id]['Crank']    = Crank(S0,strike,rf,volatility,Time, i)
    Value.loc[id]['Implicit']    = Implicit(S0, strike, rf, volatility, Time, i )
    Value.loc[id]['JR_err']   = Value.loc[id]['JR_call']-Analytical_call
    Value.loc[id]['TRG_err']  = Value.loc[id]['TRG_call']-Analytical_call
    Value.loc[id]['Crank_err']= Value.loc[id]['Crank']-Analytical_call
    Value.loc[id]['Implicit_err']= Value.loc[id]['Implicit']-Analytical_call

    
## plot the errors
plt.figure(figsize=(8 , 6))
plt.plot(Value['Steps'],Value['JR_err'],'b--',linewidth=2,label='JR Method')
plt.plot(Value['Steps'],Value['TRG_err'],'r--',linewidth=2,label='TRG Method')
plt.plot(Value['Steps'],Value['Crank_err'],'g--',linewidth=2,label='Crank Method')
plt.plot(Value['Steps'],Value['Implicit_err'],'k--',linewidth=2,label='Implicit Method')
plt.title('Relative Errors', color = 'k')
plt.xlabel('Steps')
plt.ylabel('error')
plt.legend()

#%% Question 1
plt.figure(figsize=(8 , 6))
plt.plot(Value['Steps'],Value['JR_err'],'b--',linewidth=2,label='JR Method')
plt.plot(Value['Steps'],Value['TRG_err'],'r--',linewidth=2,label='TRG Method')
plt.title('Relative Errors', color = 'k')
plt.xlabel('Steps')
plt.ylabel('error')
plt.legend()

#%% Question 2
plt.figure(figsize=(8 , 6))
plt.plot(Value['Steps'],Value['Crank_err'],'g--',linewidth=2,label='Crank Method')
plt.plot(Value['Steps'],Value['Implicit_err'],'k--',linewidth=2,label='Implicit Method')
plt.title('Relative Errors', color = 'k')
plt.xlabel('Steps')
plt.ylabel('error')
plt.legend()

#%% Question 3 Antithetic Variables Reduction Method
#Antithetic Variables Reduction Method
def monte_AV(S, K, r, sigma, T, monte, steps):
    np.random.seed(2017) # random seed 

    dt = T/steps
    nu = r - 0.5 * sigma ** 2
    # Step up Stock price matrix(change in t, monte carlo simulation) 
    # St1 and St2 are two different brownian motion
    St1, St2 = np.zeros((steps + 1, monte)),np.zeros((steps + 1, monte))
    St1[0],St2[0] = S, S
    for t in range(1, steps + 1):
        rand = np.random.standard_normal(monte) 
        St1[t] = St1[t - 1] * np.exp(nu*dt + sigma*np.sqrt(dt)*rand)
        St2[t] = St2[t - 1] * np.exp(nu*dt - sigma*np.sqrt(dt)*rand)
    
    # Call value    
    CT1 = np.exp(-r*T)*np.maximum(St1[-1] - K, 0)    
    CT2 = np.exp(-r*T)*np.maximum(St2[-1] - K, 0)   

    CT_avg = (np.sum(CT1)+np.sum(CT2))/(2*monte)
    # Covariance & Variance of Call Value
    Cov_CT = np.cov(np.stack((CT1,CT2), axis=0),ddof=1)[0,1]
    Var_CT = 0.25*(np.var(CT1,ddof=1)+np.var(CT2,ddof=1)+2*Cov_CT)
    # Call Standard Error
    SE_CT = np.sqrt(Var_CT/(2*monte))

     
    return CT_avg, SE_CT

#Control Variables Reduction Method
def monte_CV(S, K, r, sigma, T, monte, steps):
    np.random.seed(2017) # random seed 

    epsilon = 1e-5
    dt = T/steps
    nu = r - 0.5 * sigma ** 2
        
    St1 = np.zeros((steps + 1, monte))
    St1[0] = S
    CrtV = 0
    for t in range(1, steps + 1):
        TTM = (steps-t)/365
        rand = np.random.standard_normal(monte) 
        St1[t] = St1[t - 1] * np.exp(nu*dt + sigma*np.sqrt(dt)*rand)
        
        delta = BlackCallDelta(St1[t-1], K, r, sigma, T, epsilon)
        CrtV +=  delta*(St1[t] - (St1[t-1]*np.exp(r*dt)))*np.exp(r*TTM)

    
 
    CT1 = np.exp(-r*T)*( np.maximum(St1[-1] - K, 0) - CrtV )
    
    Call_Value = np.sum(CT1)/monte
    SE = np.sqrt(np.var(CT1,ddof=1)/monte)
    return Call_Value, SE

def monte_AVCV(S, K, r, sigma, T, monte, steps):
    np.random.seed(2017) # random seed 
    epsilon = 1e-5
    dt = T/steps
    nu = r - 0.5 * sigma ** 2
     
    St1, St2 = np.zeros((steps + 1, monte)),np.zeros((steps + 1, monte))
    St1[0],St2[0] = S, S
    CrtV1, CrtV2 = 0, 0
    for t in range(1, steps + 1):
        TTM = (steps-t)/365
        
        rand = np.random.standard_normal(monte) 
        St1[t] = St1[t - 1] * np.exp(nu*dt + sigma*np.sqrt(dt)*rand)
        St2[t] = St2[t - 1] * np.exp(nu*dt - sigma*np.sqrt(dt)*rand)
        
        delta1 = BlackCallDelta(St1[t-1], K, r, sigma, T, epsilon)
        CrtV1 +=  delta1*(St1[t] - (St1[t-1]*np.exp(r*dt)))*np.exp(r*TTM)
        delta2 = BlackCallDelta(St2[t-1], K, r, sigma, T, epsilon)
        CrtV2 +=  delta2*(St2[t] - (St2[t-1]*np.exp(r*dt)))*np.exp(r*TTM)
    
    # Call value    
    CT1 = np.exp(-r*T)*(np.maximum(St1[-1] - K, 0) - CrtV1)    
    CT2 = np.exp(-r*T)*(np.maximum(St2[-1] - K, 0) - CrtV2)  
    
    CT_avg = (np.sum(CT1)+np.sum(CT2))/(2*monte)
    # Covariance & Variance of Call Value
    Cov_CT = np.cov(np.stack((CT1,CT2), axis=0),ddof=1)[0,1]
    Var_CT = 0.25*(np.var(CT1,ddof=1)+np.var(CT2,ddof=1)+2*Cov_CT)
    # Call Standard Error
    SE_CT = np.sqrt(Var_CT/(2*monte))
    return CT_avg, SE_CT

n = 365
m = np.arange(500,10500,500)

#Call_price, SE1= monte_AV(S0, strike, rf, volatility, Time, 500, n)

Df_simu = pd.DataFrame(index=range(0,20),columns=['Paths','Monte_AV','AV_SE','Monte_CV','CV_SE', 'Monte_AVCV', 'AVCV_SE'])

for id,i in enumerate(m):
    #print(i)
    Df_simu.loc[id]['Paths'] = i
    Df_simu.loc[id]['Monte_AV'], Df_simu.loc[id]['AV_SE'] = monte_AV(S0, strike, rf, volatility, Time, i, n)
    Df_simu.loc[id]['Monte_CV'], Df_simu.loc[id]['CV_SE'] = monte_CV(S0, strike, rf, volatility, Time, i, n)
    Df_simu.loc[id]['Monte_AVCV'], Df_simu.loc[id]['AVCV_SE'] = monte_AVCV(S0, strike, rf, volatility, Time, i, n)

## plot the errors
plt.figure(figsize=(8 , 6))
plt.plot(Df_simu['Paths'],Df_simu['Monte_AV'],'b--',linewidth=2,label='Antithetic Reduction')
plt.plot(Df_simu['Paths'],Df_simu['Monte_CV'],'r--',linewidth=2,label='Control Reduction')
plt.plot(Df_simu['Paths'],Df_simu['Monte_AVCV'],'g--',linewidth=2,label='Antithetic & Control Reduction')
plt.title('Monte Carlo Price', color = 'k')
plt.xlabel('Steps')
plt.ylabel('Price')
plt.legend()


plt.figure(figsize=(8 , 6))
plt.plot(Df_simu['Paths'],Df_simu['AV_SE'],'b--',linewidth=2,label='Antithetic Reduction')
plt.plot(Df_simu['Paths'],Df_simu['CV_SE'],'r--',linewidth=2,label='Control Reduction')
plt.plot(Df_simu['Paths'],Df_simu['AVCV_SE'],'g--',linewidth=2,label='Antithetic & Control Reduction')
plt.ylim((0,0.1))
plt.title('Monte Standard Error', color = 'k')
plt.xlabel('Steps')
plt.ylabel('Error')
plt.legend()


