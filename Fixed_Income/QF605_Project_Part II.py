import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares, curve_fit
from scipy.stats import norm
from scipy.integrate import quad
from math import log, exp, sqrt
from scipy import interpolate
#Part II (Swaption Calibration)
#################### DD Calibration #########################################
def Black76Lognormal(F, K, T, r, sigma, opt):
    d1 = (log(F/K)+(sigma*sigma/2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    if opt == 'Call':
        return F*exp(-r*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    elif opt == 'Put':
        return K*exp(-r*T)*norm.cdf(-d2) - F*exp(-r*T)*norm.cdf(-d1)
    
def Displaced_Diffusion(F, K, T, r, sigma, beta, opt):
    return Black76Lognormal(F/beta, K+(1-beta)/beta*F, T, r, sigma*beta, opt)

def Black76ImpVol(F, K, T, r, price, opt):
    impliedVol = brentq(lambda x: price - Black76Lognormal(F, K, T, r, x, opt), 1e-6, 1)
    return impliedVol

def ImpliedVols(F, K, T, r, price, opt):
    lst = []
    for i in range(len(price)):
        lst.append(Black76ImpVol(F, K[i], T, r, price[i], opt))
    return lst

def DisDifImv(F, K1, K2, T, r, sigma, beta):
    P, C = [], []
    for k1 in K1:
        P.append(Displaced_Diffusion(F, k1, T, r, sigma, beta, 'Put'))
    for k2 in K2:
        C.append(Displaced_Diffusion(F, k2, T, r, sigma, beta, 'Call')) 
    P_imv = ImpliedVols(F, K1, T, r, P, 'Put')
    C_imv = ImpliedVols(F, K2, T, r, C, 'Call')
    return P_imv + C_imv

def DisDif_Optmize(F, K1, K2, T, r, x, MarImpVol):
    P, C = [], []
    for k1 in K1:
        P.append(Displaced_Diffusion(F, k1, T, r, x[0], x[1],'Put'))
    for k2 in K2:
        C.append(Displaced_Diffusion(F, k2, T, r, x[0], x[1],'Call'))
    P_imv = ImpliedVols(F, K1, T, r, P, 'Put')
    C_imv = ImpliedVols(F, K2, T, r, C, 'Call')
    error = np.array(P_imv + C_imv) - np.array(MarImpVol)
    return np.dot(error, error)
#%%
ExpTenor = ['1x1','1x2', '1x3','1x5','1x10',
            '5x1','5x2', '5x3','5x5','5x10',
            '10x1','10x2', '10x3','10x5','10x10']
df_DD = pd.read_excel('IR Data.xlsx', 'Swaption', skiprows=[0,1])
df_DD.reset_index(drop=True, inplace = True)
df_DD.iloc[:, 2:] = np.float64(df_DD.iloc[:, 2:]/100)
df_ForwardSwap = pd.read_csv('df_ForwardSwap.csv')
df_ForwardSwap.set_index(keys = df_ForwardSwap.columns[0], drop = True, inplace=True)
df_Strikes = df_DD.copy(deep=True)
df_DD['FSR'] = df_ForwardSwap['Swap_Rate']

strike_col = df_Strikes.columns[2:]
for i in df_Strikes.index:
    atmStrike = df_DD.loc[i, 'FSR']
    df_Strikes.loc[i, strike_col[0]]   = atmStrike-200*0.0001
    df_Strikes.loc[i, strike_col[1]]   = atmStrike-150*0.0001
    df_Strikes.loc[i, strike_col[2]]   = atmStrike-100*0.0001
    df_Strikes.loc[i, strike_col[3]]   = atmStrike-50*0.0001    
    df_Strikes.loc[i, strike_col[4]]   = atmStrike-25*0.0001
    df_Strikes.loc[i, strike_col[5]]   = atmStrike-0*0.0001
    df_Strikes.loc[i, strike_col[6]]   = atmStrike+25*0.0001
    df_Strikes.loc[i, strike_col[7]]   = atmStrike+50*0.0001
    df_Strikes.loc[i, strike_col[8]]   = atmStrike+100*0.0001
    df_Strikes.loc[i, strike_col[9]]   = atmStrike+150*0.0001
    df_Strikes.loc[i, strike_col[10]]  = atmStrike+200*0.0001
T = [1,1,1,1,1,5,5,5,5,5,10,10,10,10,10]
#%%
df_DDImpVol = pd.DataFrame()
for i in df_DD.index:
    F = df_DD['FSR'][i]
    K = df_Strikes.loc[i, '-200bps':'+200bps'].values
    r = 0        
    K1 = np.array(K)[K<=F]
    K2 = np.array(K)[K>F]

    MarketImpVol = df_DD.loc[i, '-200bps':'+200bps'].values

    w = least_squares(lambda x:DisDif_Optmize(F, K1, K2, T[i], r, x, MarketImpVol),
                      [0.2, 0.5],
                      bounds=([0.2, 1e-6],[0.4,0.9999]))
    df_DD.loc[i,'σ'] = w.x[0]
    df_DD.loc[i,'β'] = w.x[1]
    df_DDImpVol[ExpTenor[i]+' DisDifImv'] = DisDifImv(F,K1,K2,T[i],r,
               df_DD.loc[i,'σ'],df_DD.loc[i,'β'])
    print('Optimal [σ, β]for Displaced-diffusion:', [df_DD.loc[i,'σ'],df_DD.loc[i,'β']])
#%%
for i in df_Strikes.index:
    plt.figure(figsize = (8,6))
    plt.plot(df_Strikes.iloc[i, 2:],df_DDImpVol[ExpTenor[i]+' DisDifImv'],'r--',
             df_Strikes.iloc[i, 2:],df_DD.loc[i, '-200bps':'+200bps'],'bo')
    plt.show() 
#%% copy sabr.py model function
#################### SABR Calibration #########################################
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
def sabrcalibration(F, K, T, x, vols, beta):
    err = 0.0
    for i,vol in enumerate(vols):
        sabr_sol = SABR(F, K[i],T,x[0],beta,x[1],x[2])
        err +=(vol - sabr_sol)**2
    return err    
#%%    
df_SABR = pd.read_excel('IR Data.xlsx', 'Swaption', skiprows=[0,1])
df_SABR.reset_index(drop=True, inplace = True)
df_SABR.iloc[:, 2:] = np.float64(df_SABR.iloc[:, 2:]/100)
df_SABR['FSR'] = df_ForwardSwap['Swap_Rate']

SABRbeta = 0.9
initial_guess = [0.2, -0.5, 1.5]

for i in df_SABR.index:
    res = least_squares(lambda x: sabrcalibration(df_SABR.loc[i, 'FSR'],
                                                  df_Strikes.iloc[i, 2:],
                                                  T[i],
                                                  x,
                                                  df_SABR.iloc[i, 2:13],
                                                  SABRbeta),
                                                  initial_guess)
    print(ExpTenor[i],' :',res.x[0],res.x[1],res.x[2] )
    df_SABR.loc[i, 'α'] = res.x[0]
    df_SABR.loc[i, 'ρ'] = res.x[1]
    df_SABR.loc[i, 'ν'] = res.x[2]
    initial_guess = [res.x[0],res.x[1],res.x[2]]
    
df_SABRImpVol = df_SABR.copy(deep = True)
for row in df_SABRImpVol.index:
    for col in df_SABRImpVol.columns[2:13]:
        df_SABRImpVol.loc[row, col] = SABR(
                df_SABRImpVol.loc[row,'FSR'],
                df_Strikes.loc[row, col],
                T[row], 
                df_SABRImpVol.loc[row, 'α'], 
                SABRbeta,
                df_SABRImpVol.loc[row, 'ρ'],
                df_SABRImpVol.loc[row, 'ν'])
    plt.figure(figsize=(8 , 6))
    plt.plot(df_Strikes.iloc[row, 2:],df_SABRImpVol.iloc[row, 2:13],'r--',label='SABR')
    plt.plot(df_Strikes.iloc[row, 2:],df_SABR.iloc[row, 2:13],'bo',label='Market')
    plt.title(ExpTenor[row])
    plt.show()
#%% Price the following swaptions using the calibrated DD and SABR model:
df_comb = pd.read_csv('df_comb.csv')
df_comb.set_index(keys = df_comb.columns[0], drop = True, inplace=True)

df_sigma = pd.DataFrame(data    = np.array(df_DD['σ']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_beta  = pd.DataFrame(data    = np.array(df_DD['β']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_alpha = pd.DataFrame(data    = np.array(df_SABRImpVol['α']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_rho   = pd.DataFrame(data    = np.array(df_SABRImpVol['ρ']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
df_nu    = pd.DataFrame(data    = np.array(df_SABRImpVol['ν']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])

def Libor_DF(D1,D2,delta = 0.5):
    L1 = (D1/D2-1)/delta
    return L1

def PVBP(expiry,tenor,delta):
    m = int(tenor/delta)
    summa = 0
    for i in range(1,m+1):
        summa += OISDifInterp(expiry + i*delta)
    return summa*delta

def LiborDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['LIBOR_DF'])
    return np.interp(tenor, xp, yp)

def OISDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['OIS_DF'])
    return np.interp(tenor, xp, yp)

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

sigma_Pay = ParameterInterp(2,10,df_sigma)
beta_Pay  = ParameterInterp(2,10,df_beta)
alpha_Pay = ParameterInterp(2,10,df_alpha)
rho_Pay   = ParameterInterp(2,10,df_rho)
nu_Pay    = ParameterInterp(2,10,df_nu)
pvbp_Pay  = PVBP(2,10,0.5)
F_Pay = Swaption(2,10,0.5)
PricePayDD   = pvbp_Pay*Displaced_Diffusion(F_Pay,0.06,2,0,sigma_Pay,beta_Pay,'Call')
sigmaPaySABR = SABR(F_Pay, 0.06, 2, alpha_Pay, 0.9, rho_Pay, nu_Pay)
PricePaySABR = pvbp_Pay*Black76Lognormal(F_Pay, 0.06, 2, 0, sigmaPaySABR, 'Call')
    
sigma_Rec = ParameterInterp(8,10,df_sigma)
beta_Rec  = ParameterInterp(8,10,df_beta)
alpha_Rec = ParameterInterp(8,10,df_alpha)
rho_Rec   = ParameterInterp(8,10,df_rho)
nu_Rec    = ParameterInterp(8,10,df_nu)
pvbp_Rec = PVBP(8,10,0.5)
F_REC = Swaption(8,10,0.5)
PriceRecDD   = pvbp_Rec*Displaced_Diffusion(F_REC,0.02,8,0,sigma_Rec,beta_Rec,'Put')
sigmaRecSABR = SABR(F_REC, 0.02, 8, alpha_Rec, 0.9, rho_Rec, nu_Rec)
PriceRecSABR = pvbp_Rec*Black76Lognormal(F_REC, 0.02, 8, 0, sigmaRecSABR, 'Put')

print('[PricePayDD, PricePaySABR]:',[PricePayDD,PricePaySABR])
print('[PriceRecDD, PriceRecSABR]:',[PriceRecDD,PriceRecSABR])

# export for part II
df_SABRImpVol.to_csv('df_SABRImpVol.csv', sep=',')
