import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.stats import norm
from scipy.integrate import quad
from math import log, exp, sqrt
from scipy import interpolate

# Part III (Convexity Correction)
df_SABRImpVol = pd.read_csv('df_SABRImpVol.csv')
df_SABRImpVol.set_index(keys = df_SABRImpVol.columns[0], drop = True, inplace=True)
df_comb = pd.read_csv('df_comb.csv')
df_comb.set_index(keys = df_comb.columns[0], drop = True, inplace=True)
df_swaption = pd.read_csv('df_swaption.csv')
df_swaption.set_index(keys = df_swaption.columns[0], drop = True, inplace=True)
#%%
def Black76Lognormal(F, K, T, r, sigma, opt):
    d1 = (log(F/K)+(sigma*sigma/2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    if opt == 'Call':
        return F*exp(-r*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    elif opt == 'Put':
        return K*exp(-r*T)*norm.cdf(-d2) - F*exp(-r*T)*norm.cdf(-d1)
    
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
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

def Libor_DF(D1,D2,delta = 0.5):
    L1 = (D1/D2-1)/delta
    return L1

def LiborDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['LIBOR_DF'])
    return np.interp(tenor, xp, yp)

def OISDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['OIS_DF'])
    return np.interp(tenor, xp, yp)

def PVBP(expiry,tenor,delta):
    m = int(tenor/delta)
    summa = 0
    for i in range(1,m+1):
        summa += OISDifInterp(expiry + i*delta)
    return summa*delta

def floatLeg(expiry,tenor,delta):
    m = int(tenor/delta)
    summa = 0
    for i in range(1,m+1):
        loc = expiry + i*delta
        summa += OISDifInterp(loc)*Libor_DF(LiborDifInterp(loc-delta),LiborDifInterp(loc),delta)
    return summa*delta

def Swaption(expiry, tenor, delta):
    return floatLeg(expiry,tenor,delta)/PVBP(expiry,tenor,delta)

def ParameterInterp(expiry,tenor,df):
   x = df.columns
   y = df.index
   z = df.values
   f = interpolate.interp2d(x, y, z, kind='linear')
   return f(tenor, expiry)[0]

df_alpha = pd.DataFrame(data    = np.array(df_SABRImpVol['α']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_rho   = pd.DataFrame(data    = np.array(df_SABRImpVol['ρ']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_nu    = pd.DataFrame(data    = np.array(df_SABRImpVol['ν']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])

def IRR(tenor, delta, K):
    summa = 0
    for i in range(1,tenor+1):
        summa += delta*(1+delta*K)**(-i)
    return summa

def IRR1st(tenor, delta, K):
    summa = 0
    for i in range(1,tenor+1):
        summa += (delta**2)*(-i)*(1+delta*K)**(-i-1)
    return summa

def IRR2nd(tenor, delta, K):
    summa = 0
    for i in range(1,tenor+1):
        summa += (delta**3)*(-i)*(-i-1)*(1+delta*K)**(-i-2)
    return summa

def h2nd(tenor, delta, K):
    IRR0 = IRR(tenor, delta, K)
    IRR1 = IRR1st(tenor, delta, K)
    IRR2 = IRR2nd(tenor, delta, K) 
    return (-IRR2*K - 2*IRR1)/IRR0**2 + (2*IRR1**2*K)/IRR0**3

def integrand(SnN0, K, T, r, sigma, tenor, delta, opt):
    irr = IRR(tenor, delta, SnN0)
    hppk = h2nd(tenor, delta, K)
    return hppk*irr*Black76Lognormal(SnN0, K, T, r, sigma, opt)

def CMS(expiry, tenor, delta):
    SnN0    = Swaption(expiry, tenor, delta)
    alpha   = ParameterInterp(expiry,tenor,df_alpha)
    rho     = ParameterInterp(expiry,tenor,df_rho)
    nu      = ParameterInterp(expiry,tenor,df_nu)
    Rec     = quad(lambda x:integrand(SnN0,x,expiry,0,
                                  SABR(SnN0,x,expiry,alpha,0.9,rho,nu),
                                  tenor, delta, 'Put'), 0, SnN0)
    Pay     = quad(lambda x:integrand(SnN0,x,expiry,0,
                                  SABR(SnN0,x,expiry,alpha,0.9,rho,nu),
                                  tenor, delta, 'Call'), SnN0, 0.1)
    return SnN0 + Rec[0] + Pay[0]

def PV_CMS(tenor, T, delta):
    n = int(T/delta)
    pv = 0
    for i in range(1,n+1):
        loc = i*delta
        pv += OISDifInterp(loc)*delta*CMS(loc,tenor,delta)
    return pv
#%%
df_CMS = df_swaption.copy(deep = True)
for i in df_CMS.index:
    for j in df_CMS.columns:
        df_CMS.loc[i,j] = CMS(int(i),int(j),0.5)
print(df_swaption)
print(df_CMS)

for i in df_swaption.index:
    plt.figure(figsize=(6,4))
    plt.plot([1,2,3,5,10], df_swaption.loc[i,:],'r',[1,2,3,5,10], df_CMS.loc[i,:],'b')
    plt.legend(['Forward Swap Rate','CMS'])
    plt.title(str(i) + 'Y Expiry')
    plt.xlabel('Tenor')
    plt.show()
    
pv_CMS10y = PV_CMS(10, 5, 0.5)
pv_CMS2y = PV_CMS(2, 10, 0.25)
print('[pv_CMS10y, pv_CMS2y]:', [pv_CMS10y, pv_CMS2y])

