# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:28:35 2017

@author: Jon Wee
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad 
import pandas as pd
import datetime as dt
from math import log, exp, sqrt, pi
from Final_Project_II import S0, r, T, ATMvol ,Diff_sigma, F, df_comb, SABRbeta, rho, alpha, nu    
#from scipy.optimize import brentq, curve_fit, least_squares
#from math import log, exp, sqrt

# 1. Black Scholes Model
def BlackScholes(S, r, T, sigma):
    num1 = (S**2)*exp(2*r*T+(sigma**2)*T)
    num2 = 5*sqrt(S)*exp(0.5*r*T-(0.125*sigma**2)*T)
    num3 = 100
    V0 = np.exp(-r*T)*(num1+num2+num3)
    return V0

print("Black Scholes Model European Contract price is: ", BlackScholes(S0, r, T, ATMvol))

# 2. Bachelier Model
def Bachelier(S, r, T, sigma):
    num1 = S**2*(1+sigma**2*T)
    num2 = quad(lambda x : 1/sqrt(2*pi)*sqrt(1+sigma*x*(sqrt(T)))*(exp(-x**2/2)), 0, 5000)
    num2_ =  5*sqrt(S)*num2[0]
    num3 = 100
    V0 = exp(-r*T)*(num1+num2_+num3)
    return V0

print("Bachelier Model European Contract price is: ", Bachelier(S0, 0, T, ATMvol))
# Using r=0 for Bachelier model

# 3. Static Replication

"""  
# sigma = 0.4, T=1, E_Var becomes 0.16#
I_call = quad(lambda x: callintegrand(x, S0, r, 1, 0.4), F, 50000)
I_put  = quad(lambda x: putintegrand(x, S0, r, 1, 0.4),0.0, F)
E_var = 2*exp(r*T)*(I_put[0]+I_call[0])
print (E_var )
"""

def Black76Lognormalcall(F, K, r, sigma, T):
    d1 = (log(F/K)+(0.5*(sigma**2)*T))/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Lognormalput(F, K, r, sigma, T):
    d1 = (log(F/K)+(0.5*(sigma**2)*T))/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F is K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F*(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

def callintegrand(K, F, r, T, sigma):
    price = Black76Lognormalcall(F, K, r, sigma, T)*h_dd(K)
    return price

def putintegrand(K, F, r, T, sigma):
    price = Black76Lognormalput(F, K, r, sigma, T)*h_dd(K)
    return price

def h_dd(K):
    return 2 - 1.25*(K**(-1.5))

#I_call = quad(lambda x: callintegrand(x, S0, r, 1, 0.4), F, 50000)
I_call = quad(lambda kk: callintegrand(kk, F, r, T, SABR(F, kk, T, alpha, SABRbeta, rho, nu)) , F, 5000)
I_put  = quad(lambda kk: putintegrand(kk, F, r, T, SABR(F, kk, T, alpha, SABRbeta, rho, nu)) , 0.0, F)

staticprice = exp(-r*T)*(F**2 + 5*F**0.5 + 100) + (I_call[0]+I_put[0])
print ("Static Replication European Contract price is: ", staticprice )
