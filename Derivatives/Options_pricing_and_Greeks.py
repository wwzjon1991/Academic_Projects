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


    
    
    
    