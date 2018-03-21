# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:04:39 2018

@author: Jon Wee
"""

import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from math import log, exp, sqrt, pi

#%% Functions
def Black76Lognormalcall(F, K, r, sigma, T):
    d1 = (log(F/K)+(0.5*(sigma**2)*T))/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Lognormalput(F, K, r, sigma, T):
    d1 = (log(F/K)+(0.5*(sigma**2)*T))/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

####################################################################################################################################
############   GREEKS    ###########################################################################################################

def BlackCallDelta(F, K, r, sigma, T, epsilon):
    return (Black76Lognormalcall(F+epsilon, K, r, sigma, T) - Black76Lognormalcall(F-epsilon, K, r, sigma, T)) / (2 * epsilon)

def BlackCallGamma(F, K, r, sigma, T, epsilon):
    return (Black76Lognormalcall(F+epsilon, K, r, sigma, T) - 2* Black76Lognormalcall(F, K, r, sigma, T)\
            + Black76Lognormalcall(F-epsilon, K, r, sigma, T)) / (2 * epsilon)

def BlackCallVega(F, K, r, sigma, T, epsilon):
    return (Black76Lognormalcall(F, K, r, sigma+epsilon, T) - Black76Lognormalcall(F, K, r, sigma-epsilon, T)) / (2 * epsilon)

def BlackCallTheta(F, K, r, sigma, T, epsilon):
    return (Black76Lognormalcall(F, K, r, sigma, T-epsilon) - Black76Lognormalcall(F, K, r, sigma, T)) / (2 * epsilon)

def BlackPutTheta(F, K, r, sigma, T, epsilon):
    return (Black76Lognormalput(F, K, r, sigma, T-epsilon) - Black76Lognormalput(F, K, r, sigma, T)) / (2 * epsilon)


#%% European CAll/Put Options Pricing Greeks

Epsilon  = 1e-4      # Need to Calibrate Epsilon, different Epsilon gives different greeks. 
rf = 0.01
Time = 1.0
Sigma = 0.20
S0 = np.linspace(1, 100, 101)
Strike = 50

F = S0*exp(rf*Time) # Forward 
n = len(F)

OptionPricing = pd.DataFrame(index=range(0,101),columns=['Call', 'Put'])

Greeks = pd.DataFrame(index=range(0,101),columns=['Delta_Call', 'Delta_Put', 'Gamma_Call_Put', 'Vega_Call_Put', 'Theta_Call', 'Theta_Put'])

for id in range(n):
    OptionPricing.loc[id]['Call'] = Black76Lognormalcall(S0[id], Strike, rf, Sigma, Time)
    OptionPricing.loc[id]['Put']  = Black76Lognormalput(S0[id], Strike, rf, Sigma, Time)
    ###################             GREEKS              #########################################################################
    Greeks.loc[id]['Delta_Call'] = BlackCallDelta(S0[id], Strike, rf, Sigma, Time, Epsilon)    # Call Delta
    Greeks.loc[id]['Delta_Put']  = BlackCallDelta(S0[id], Strike, rf, Sigma, Time, Epsilon)-1  # Put Delta
    Greeks.loc[id]['Gamma_Call_Put']  = BlackCallGamma(S0[id], Strike, rf, Sigma, Time, Epsilon) # Gamma of Call/Put
    Greeks.loc[id]['Vega_Call_Put']  = BlackCallVega(S0[id], Strike, rf, Sigma, Time, Epsilon) # Vega of Call/Put
    Greeks.loc[id]['Theta_Call'] = BlackCallTheta(S0[id], Strike, rf, Sigma, Time, Epsilon)    # Call Theta
    Greeks.loc[id]['Theta_Put']  = BlackPutTheta(S0[id], Strike, rf, Sigma, Time, Epsilon)    # Put Theta

plt.figure(figsize=(8 , 6))
plt.plot(S0,OptionPricing['Call'],'b--',linewidth=2,label='Call')
plt.title('Option Pricing', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Price')
plt.legend()

plt.figure(figsize=(8 , 6))
plt.plot(S0,OptionPricing['Put'],'r--',linewidth=2,label='Put')
plt.title('Option Pricing', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Price')
plt.legend()

####################################################################################################################################
############   GREEKS    ###########################################################################################################

plt.figure(figsize=(12 , 8))
plt.plot(S0,Greeks['Delta_Call'],'r--',linewidth=2,label='Delta_Call')
plt.title('Option Greeks: Delta', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Delta')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Greeks['Delta_Put'],'r--',linewidth=2,label='Delta_Put')
plt.title('Option Greeks: Delta', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Delta')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Greeks['Gamma_Call_Put'],'r--',linewidth=2,label='Gamma')
plt.title('Option Greeks: Gamma', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Gamma')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Greeks['Vega_Call_Put'],'r--',linewidth=2,label='Vega')
plt.title('Option Greeks: Vega', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Vega')
plt.legend()


plt.figure(figsize=(12 , 8))
plt.plot(S0,Greeks['Theta_Call'],'r--',linewidth=2,label='Theta_Call')
plt.plot(S0,Greeks['Theta_Put'],'b--',linewidth=2,label='Theta_Put')
plt.title('Option Greeks: Theta', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Theta')
plt.legend()

#%%   Call Spread Options
def BullSpread(F, K, r, sigma, T, spread):
    LongCall  = Black76Lognormalcall(F, K-spread, r, sigma, T)
    ShortCall = -Black76Lognormalcall(F, K+spread, r, sigma, T)
    return (LongCall+ShortCall) / (2*spread)

def BearSpread(F, K, r, sigma, T, spread):
    LongPut  = Black76Lognormalput(F, K+spread, r, sigma, T)
    ShortPut = -Black76Lognormalput(F, K-spread, r, sigma, T)
    return (LongPut+ShortPut) / (2*spread)    
    
def BullDelta(F, K, r, sigma, T, spread, epsilon):
    return (BullSpread(F+epsilon, K, r, sigma, T, spread) - BullSpread(F-epsilon, K, r, sigma, T, spread)) / (2 * epsilon)

def BearDelta(F, K, r, sigma, T, spread, epsilon):
    return (BearSpread(F+epsilon, K, r, sigma, T, spread) - BearSpread(F-epsilon, K, r, sigma, T, spread)) / (2 * epsilon)


def BullGamma(F, K, r, sigma, T, spread, epsilon):
    return (BullSpread(F+epsilon, K, r, sigma, T, spread) - 2* BullSpread(F, K, r, sigma, T, spread)\
            + BullSpread(F-epsilon, K, r, sigma, T, spread)) / (2 * epsilon)

def BullVega(F, K, r, sigma, T, spread, epsilon):
    return (BullSpread(F, K, r, sigma+epsilon, T, spread) - BullSpread(F, K, r, sigma-epsilon, T, spread)) / (2 * epsilon)



Epsilon  = 1e-4      # Need to Calibrate Epsilon, different Epsilon gives different greeks. 
rf = 0.01
Time = 1.0
Sigma = 0.20
S0 = np.linspace(1, 100, 101)
Strike = 50
Spread = 5

F = S0*exp(rf*Time) # Forward 
n = len(F)

Replica = pd.DataFrame(index=range(0,101),columns=['Bull_Spread', 'Bear_Spread', 'Delta_Call', 'Delta_Put', 'Gamma_Call_Put', 'Vega_Call_Put'])

for id in range(n):
    Replica.loc[id]['Bull_Spread']  = BullSpread(S0[id], Strike, rf, Sigma, Time, Spread)
    Replica.loc[id]['Bear_Spread']  = BearSpread(S0[id], Strike, rf, Sigma, Time, Spread)
    ###################             GREEKS              #########################################################################    
    Replica.loc[id]['Delta_Call']      = BullDelta(S0[id], Strike, rf, Sigma, Time, Spread, Epsilon)    # Call Delta
    Replica.loc[id]['Delta_Put']       = BearDelta(S0[id], Strike, rf, Sigma, Time, Spread, Epsilon)  # Put Delta
    Replica.loc[id]['Gamma_Call_Put']  = BullGamma(S0[id], Strike, rf, Sigma, Time, Spread, Epsilon) # Gamma of Call/Put
    Replica.loc[id]['Vega_Call_Put']   = BullVega(S0[id],  Strike, rf, Sigma, Time, Spread, Epsilon) # Vega of Call/Put

plt.figure(figsize=(8 , 6))
plt.plot(S0,Replica['Bull_Spread'],'b--',linewidth=2,label='Bull_Spread')
plt.title('Option Pricing: Bull_Spread', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Price')
plt.legend()    
    
plt.figure(figsize=(8 , 6))
plt.plot(S0,Replica['Bear_Spread'],'b--',linewidth=2,label='Bear_Spread')
plt.title('Option Pricing: Bear_Spread', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Price')
plt.legend()    
        
####################################################################################################################################
############   GREEKS    ###########################################################################################################

plt.figure(figsize=(12 , 8))
plt.plot(S0,Replica['Delta_Call'],'r--',linewidth=2,label='Delta_Call')
plt.title('Option Greeks: Delta', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Delta')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Replica['Delta_Put'],'r--',linewidth=2,label='Delta_Put')
plt.title('Option Greeks: Delta', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Delta')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Replica['Gamma_Call_Put'],'r--',linewidth=2,label='Gamma')
plt.title('Option Greeks: Gamma', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Gamma')
plt.legend()

plt.figure(figsize=(12 , 8))
plt.plot(S0,Replica['Vega_Call_Put'],'r--',linewidth=2,label='Vega')
plt.title('Option Greeks: Vega', color = 'k')
plt.xlabel('Strikes')
plt.ylabel('Vega')
plt.legend()


#%% European options 
# Based on TRG binomial Tree
def EuroCall(S, K, r, sigma, T, steps):
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
        
    Call_value = Call[0,0]
    return Call_value
    
def EuroPut(S, K, r, sigma, T, steps):
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
        
    Put = np.zeros((steps+1,steps+1))
    Put[steps,:]=np.maximum(0, K-St)
    
    DF = np.exp(-r*dt)
    for i in range(steps-1,0-1,-1):
        #print                          up           ddown          
        Put[i,0:i+1] =  DF*(P_u*Put[i+1,:i+1]+P_d*Put[i+1,1:i+2])
        
    Put_value = Put[0,0]
    return Put_value


#%% American options

def AmericanCall(S, K, r, sigma, T, steps):
    dt = T/steps
    nu = r-0.5*(sigma**2)
    x = np.log(S)
    x_delta = np.sqrt((sigma**2)*dt + (nu**2)*(dt**2))
        
    P_u = 0.5 + 0.5*(nu*dt/x_delta)
    P_d = 0.5 - 0.5*(nu*dt/x_delta)
    # stock price at maturity
    St = np.zeros((steps+1,steps+1))
    #expectd stock prices at time T
    for i in range(0,steps+1):
        St[i,-1] = np.exp(x_delta*(steps-i)-(x_delta*(i))+x)
            
    Call = np.zeros((steps+1,steps+1))
    Call[:,steps]=np.maximum(0, St[:,-1]-K)
    
    DF = np.exp(-r*dt)
    
    for i in range(steps, 0,-1):
        #    print(i)
        #    for j in range(i):
        #        print(j,i-1)
        Call[:i , i-1] =   DF*(P_u*Call[:i , i] + P_d*Call[1:i+1 , i])
        #     print(Call[:i, i-1])
        St[:i , i-1]   =   St[:i , i] * np.exp(-x_delta) 
        #     print(St[:i, i-1])
        Call[:i , i-1] = np.maximum(St[:i , i-1] - K, Call[:i , i-1])
            
    Call_Value = Call[0,0]

    return Call_Value 

def AmericanPut(S, K, r, sigma, T, steps):
    dt = T/steps
    nu = r-0.5*(sigma**2)
    x = np.log(S)
    x_delta = np.sqrt((sigma**2)*dt + (nu**2)*(dt**2))
        
    P_u = 0.5 + 0.5*(nu*dt/x_delta)
    P_d = 0.5 - 0.5*(nu*dt/x_delta)
    # stock price at maturity
    St = np.zeros((steps+1,steps+1))
    #expectd stock prices at time T
    for i in range(0,steps+1):
        St[i,-1] = np.exp(x_delta*(steps-i)-(x_delta*(i))+x)
            
    Put = np.zeros((steps+1,steps+1))
    Put[:,steps]=np.maximum(0, K-St[:,-1])
    
    DF = np.exp(-r*dt)
    
    for i in range(steps, 0,-1):
        Put[:i , i-1] =   DF*(P_u*Put[:i , i] + P_d*Put[1:i+1 , i])
        St[:i , i-1]   =   St[:i , i] * np.exp(-x_delta) 
        Put[:i , i-1] = np.maximum(K - St[:i , i-1] , Put[:i , i-1])
            
    Put_Value = Put[0,0]
    return Put_Value 

#%% Barrier options 
# Call option for Up-Out and Up-In | Down-Out and Down-In

def Up_Out(S, K, r, sigma, T, steps, H):
    dt = T/steps
    nu = r-0.5*(sigma**2)
    x = np.log(S)
    x_delta = np.sqrt((sigma**2)*dt + (nu**2)*(dt**2))
        
    P_u = 0.5 + 0.5*(nu*dt/x_delta)
    P_d = 0.5 - 0.5*(nu*dt/x_delta)
    # stock price at maturity
    St = np.zeros((steps+1,steps+1))
    #expectd stock prices at time T
    for i in range(0,steps+1):
        St[i,-1] = np.exp(x_delta*(steps-i)-(x_delta*(i))+x)
            
    Knock_Up_Out = np.zeros((steps+1,steps+1))
    for i in range(steps+1):
        if St[i , -1] < H: # below barrier then got call value 
            Knock_Up_Out[i , -1] = np.maximum(0 , St[i , -1] - K)
        else: # Above H==0
            Knock_Up_Out[i , -1] = 0


    DF = np.exp(-r*dt)
    
    for i in range(steps, 0,-1): # Column
        #    print(i)
        for j in range(i): # Row
            St[j , i-1] = St[j , i-1] * np.exp(-x_delta) 
            if St[j , i-1] < H: # below barrier then got call value 
                Knock_Up_Out[j , i-1] = DF*(P_u*Knock_Up_Out[j , i]+P_d*Knock_Up_Out[j+1 , i])
                Knock_Up_Out[j , i-1] = np.maximum(Knock_Up_Out[j , i-1], St[j , i-1]-K)
            else: # Above H==0
                Knock_Up_Out[j , i-1] = 0

    Knock_Up_Out_Value = Knock_Up_Out[0,0]
    return Knock_Up_Out_Value

def Up_In(S, K, r, sigma, T, steps, H):
    Up_In_Value = EuroCall(S, K, r, sigma, T, steps) - Up_Out(S, K, r, sigma, T, steps, H)
    return Up_In_Value

def Down_In(S, K, r, sigma, T, steps, H):
    dt = T/steps
    nu = r-0.5*(sigma**2)
    x = np.log(S)
    x_delta = np.sqrt((sigma**2)*dt + (nu**2)*(dt**2))
        
    P_u = 0.5 + 0.5*(nu*dt/x_delta)
    P_d = 0.5 - 0.5*(nu*dt/x_delta)
    # stock price at maturity
    St = np.zeros((steps+1,steps+1))
    #expectd stock prices at time T
    for i in range(0,steps+1):
        St[i,-1] = np.exp(x_delta*(steps-i)-(x_delta*(i))+x)
            
    Knock_Down_In = np.zeros((steps+1,steps+1))
    for i in range(steps+1):
        if St[i , -1] < H: # below barrier then got call value 
            Knock_Down_In[i , -1] = np.maximum(0 , St[i , -1] - K)
        else:
            Knock_Down_In[i , -1] = 0


    DF = np.exp(-r*dt)
    
    for i in range(steps, 0,-1): # Column
        #    print(i)
        for j in range(i): # Row
            St[j , i-1] = St[j , i-1] * np.exp(-x_delta) 
            if St[j , i-1] < H: # below barrier then got call value 
                Knock_Down_In[j , i-1] = DF*(P_u*Knock_Down_In[j , i]+P_d*Knock_Down_In[j+1 , i])
                Knock_Down_In[j , i-1] = np.maximum(Knock_Down_In[j , i-1], St[j , i-1]-K)
            else:
                Knock_Down_In[j , i-1] = 0

    Knock_Down_In_Value = Knock_Down_In[0,0]
    return Knock_Down_In_Value

def Down_Out(S, K, r, sigma, T, steps, H):
    Down_Out_Value = EuroCall(S, K, r, sigma, T, steps) - Down_In(S, K, r, sigma, T, steps, H)
    return Down_Out_Value

#%%     
rf = 0.03
Time = 1
volatility = 0.25
S0 = 30
strike = 31
n = 5
H = 35

print("European Call: ", EuroCall(S0, strike, rf, volatility, Time, n ))
print("European Put: ", EuroPut(S0, strike, rf, volatility, Time, n ))    
print("American Call: ", AmericanCall(S0, strike, rf, volatility, Time, n )) 
print("American Put: ", AmericanPut(S0, strike, rf, volatility, Time, n ))   
print("Up-Out Call: ", Up_Out(S0, strike, rf, volatility, Time, n, H))   
print("Up-In Call: ", Up_In(S0, strike, rf, volatility, Time, n, H ))   
