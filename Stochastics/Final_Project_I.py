# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:29:26 2017

@author: Jon Wee
"""

import numpy as np
from scipy.stats import norm


#############################################################################################
""" START Function for the 4 types of Models  """
'''
S == Stock Price = 100
F == Forwards = 100
K == Strike Price = 102
beta == BETA - risk = 1
r = risk free i/r = 0.01
sigma == volitily = 0.12
T == Time period = 0.5
opt == option type (0=call,1=put,2=Cash_call, 3=Cash_put,4=Asset_call, 5=Asset_put)
'''
######################################################################################################
# VALUES
"""
stock = 100  # Stock Price
forward  = 100  # Forwards
strike  = 102 # Strike Price
beta = 0.8
interest_rate = 0.01  # Interest rate
BACH_RATE = 0
sigma = 0.12
time  = 0.5 # Time
"""
# stock,forward,strike,beta,interest_rate,sigma,time
# stock,strike,interest_rate,sigma,time
# forward,strike,interest_rate,sigma,time
######################################################################################################
def Black_Scholes( S, K, r, sigma, T, opt) :

    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 0: # call option
        call_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_value
    elif opt == 1: # put option
        put_value = -S*norm.cdf(-d1) + K*np.exp(-r*T)*norm.cdf(-d2)
        return put_value
    elif opt == 2: # digital cash or nothing call option
        Cash_call_value = np.exp(-r*T)*norm.cdf(d2)
        return Cash_call_value
    elif opt == 3:
        Cash_put_value = np.exp(-r*T)*norm.cdf(-d2)
        return Cash_put_value
    elif opt == 4:
        Asset_call_value = S*norm.cdf(d1)
        return Asset_call_value    
    elif opt == 5:
        Asset_put_value = S*norm.cdf(-d1)
        return Asset_put_value
"""
print("######### BLACK SCHOLES MODEL ##################")      
print('Black Scholes call options:', Black_Scholes(stock,strike,interest_rate,sigma,time,0))
print('Black Scholes put options:', Black_Scholes(stock,strike,interest_rate,sigma,time,1))
print('Black Scholes digital cash or nothing call options:', Black_Scholes(stock,strike,interest_rate,sigma,time,2))
print('Black Scholes digital cash or nothing put options:', Black_Scholes(stock,strike,interest_rate,sigma,time,3))
print('Black Scholes digital asset or nothing call options:', Black_Scholes(stock,strike,interest_rate,sigma,time,4))
print('Black Scholes digital asset or nothing put options:', Black_Scholes(stock,strike,interest_rate,sigma,time,5))
print(" ")
"""
#################################################################################################################

def Bachelier(S, K, r, sigma, T, opt):
    
    d1 = (S-K)/(sigma*S*(T**0.5))
#   d1 = S/(sigma*(T**0.5)) - K/(sigma*(T**0.5))
    if opt == 0: # call option
        call_value = np.exp(-r*T)*((S-K)*norm.cdf(d1) + (sigma*S*(T**0.5)*norm.pdf(d1)))
        return call_value
    elif opt == 1: # put option
        put_value = np.exp(-r*T)*((K-S)*norm.cdf(-d1) + (sigma*S*(T**0.5)*norm.pdf(d1)))
        return put_value
    elif opt == 2:
        Cash_call_value = np.exp(-r*T)*norm.cdf(d1)
        return Cash_call_value
    elif opt == 3:
        Cash_put_value = np.exp(-r*T)*norm.cdf(-d1)
        return Cash_put_value
    elif opt == 4: # Digital Asset or nothing call
        Asset_call_value = S*np.exp(-r*T)*(norm.cdf(d1)+sigma*(T**0.5)*norm.pdf(-d1))
        return Asset_call_value
    elif opt == 5:  # Digital Asset or nothing put
        Asset_put_value =  S*np.exp(-r*T)*(norm.cdf(-d1)-sigma*(T**0.5)*norm.pdf(d1))
        return Asset_put_value
"""
print("######### BACHELIER MODEL ##################")  
print('Bachelier model call options:', Bachelier(stock,strike,BACH_RATE,sigma,time,0))
print('Bachelier model put options:', Bachelier(stock,strike,BACH_RATE,sigma,time,1))
print('Bachelier model digital cash or nothing call options:', Bachelier(stock,strike,BACH_RATE,sigma,time,2))
print('Bachelier model digital cash or nothing put options:', Bachelier(stock,strike,BACH_RATE,sigma,time,3))
print('Bachelier model digital asset or nothing call options:', Bachelier(stock,strike,BACH_RATE,sigma,time,4))
print('Bachelier model digital asset or nothing call options:', Bachelier(stock,strike,BACH_RATE,sigma,time,5))
print(" ")
"""
############################################################################################################
def Black76(F, K, r, sigma, T, opt):
    
#    d1 = (np.log(F/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d1 = (np.log(F/K) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 0: # call option
        call_value = F*np.exp(-r*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_value
    elif opt == 1: # put option
        put_value = -F*np.exp(-r*T)*norm.cdf(-d1) + K*np.exp(-r*T)*norm.cdf(-d2)
        return put_value
    elif opt == 2: # Cash call
        Cash_call_value = np.exp(-r*T)*norm.cdf(d2)
        return Cash_call_value
    elif opt == 3: # Cash put
        Cash_put_value = np.exp(-r*T)*norm.cdf(-d2)
        return Cash_put_value
    elif opt == 4: # Asset call
        Asset_call_value = F*np.exp(-r*T)*norm.cdf(d1)
        return Asset_call_value    
    elif opt == 5: # Asset put
        Asset_put_value = F*np.exp(-r*T)*norm.cdf(-d1)
        return Asset_put_value
"""
print("######### BLACK76 MODEL ##################")  
print('Black76 model call options:', Black76(forward,strike,interest_rate,sigma,time,0))
print('Black76 model put options:', Black76(forward,strike,interest_rate,sigma,time,1))
print('Black76 model digital cash or nothing call options:', Black76(forward,strike,interest_rate,sigma,time,2))
print('Black76 model digital cash or nothing put options:', Black76(forward,strike,interest_rate,sigma,time,3))
print('Black76 model digital asset or nothing call options:', Black76(forward,strike,interest_rate,sigma,time,4))
print('Black76 model digital asset or nothing call options:', Black76(forward,strike,interest_rate,sigma,time,5))
print(" ")
"""
####################################################################################################

def Displaced_Diffusion(F, K, beta, r, sigma, T, opt):
    Fd = F/beta
    Kd = K + ((1-beta)/beta)*F
    sigma_d = sigma*beta
    
    d1 = (np.log(Fd/Kd) + (0.5*sigma_d**2)*T)/(sigma_d*np.sqrt(T))
    d2 = d1 - sigma_d*np.sqrt(T)
    
    if opt == 0: # call option
        call_value = Fd*np.exp(-r*T)*norm.cdf(d1) - Kd*np.exp(-r*T)*norm.cdf(d2)
        return call_value
    elif opt == 1: # put option
        put_value = -Fd*np.exp(-r*T)*norm.cdf(-d1) + Kd*np.exp(-r*T)*norm.cdf(-d2)
        return put_value
    elif opt == 2:
        Cash_call_value = np.exp(-r*T)*norm.cdf(d2)
        return Cash_call_value
    elif opt == 3:
        Cash_put_value = np.exp(-r*T)*norm.cdf(-d2)
        return Cash_put_value
    elif opt == 4:
        Asset_call_value = Fd*np.exp(-r*T)*norm.cdf(d1)
        return Asset_call_value    
    elif opt == 5:
        Asset_put_value = Fd*np.exp(-r*T)*norm.cdf(-d1)
        return Asset_put_value
"""       
print("######### DISPLACED DIFFUSION MODEL ##################")        
print('Displaced_Diffusion model call options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,0))
print('Displaced_Diffusion model put options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,1))
print('Displaced_Diffusion model digital cash or nothing call options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,2))
print('Displaced_Diffusion model digital cash or nothing put options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,3))
print('Displaced_Diffusion model digital asset or nothing call options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,4))
print('Displaced_Diffusion model digital asset or nothing call options:', Displaced_Diffusion(forward,strike,beta,interest_rate,sigma,time,5))
print(" ")    
print("############# END END END END END #####################")   
"""    
  

#""" Function for the 4 types of Models  END """
#####################################################################################################################
