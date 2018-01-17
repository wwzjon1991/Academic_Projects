# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 01:06:23 2017

@author: Jon Wee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Data files
industrial_portfolios = 'Industry_Portfolios.xlsx'
Market_Returns = 'Market_Returns.xlsx'
Risk_Factors = 'Risk_Factors.xlsx'

# columns names
Ind_cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops','Hlth', 'Utils', 'Other']


# Industrial Portfolios
Data_Ind = pd.read_excel(industrial_portfolios)
Data_Ind = Data_Ind.drop('Date', 1)

# Market
Data_Mkt = pd.read_excel(Market_Returns)
Data_Mkt = Data_Mkt.drop('Date', 1)

# Factors
Data_Factors = pd.read_excel(Risk_Factors)
Data_Factors = Data_Factors.drop('Date', 1)

# Risk Free
Data_Rf = pd.DataFrame(Data_Factors['Rf'])

# Combine Data 
Data_assets = pd.concat([Data_Ind, Data_Mkt,Data_Factors], axis=1)

# Sharpe or Information ratio
def Sharpe(ind, rf):
    excess_ret = np.array(ind)-np.array(rf)
    sharpe = []
    for i in range(10):
        x = excess_ret[:,i].mean()/excess_ret[:,i].std(ddof=1)
        sharpe = np.append(sharpe,x)
    return sharpe

Sharpe = Sharpe(Data_Ind, Data_Rf)

# Sortino
def Sortino(ind, rf):
    rf= np.array(rf)
    ind = np.array(ind) 
    excess_ret = ind-rf
    min = np.minimum(0 , excess_ret)

    downside_risk = (min**2)
    Sortino = []
    for i in range(10):
        Sortino = np.append(Sortino, np.mean(excess_ret[:,i])/np.sqrt(downside_risk[:,i].mean()))
    return Sortino

Sortino = Sortino(Data_Ind,Data_Rf)

# Treynor
def Treynor(ind, rf, mkt):
    rf = np.array(rf)
    ind = np.array(ind)
    excess_ret = ind-rf
    beta = []
    x = np.array(mkt)-rf
    x = sm.add_constant(x)
    y = excess_ret
    # beta regression
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        beta = np.append(beta, results.params[1])    
    treynor = []
    for i in range(10):
        x = np.mean(excess_ret[:,i])/beta[i]
        treynor = np.append(treynor, x) 
    
    return treynor

Treynor = Treynor(Data_Ind, Data_Rf, Data_Mkt)

def Jensen(ind, rf, mkt):
    rf = np.array(rf)
    ind = np.array(ind)
    mkt = np.array(mkt)
    x = mkt-rf
    x = sm.add_constant(x)
    y = ind-rf
    
    alpha = []
    
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        alpha = np.append(alpha, results.params[0])
    return alpha

Jensen = Jensen(Data_Ind, Data_Rf, Data_Mkt)

def carhart_4factor(ind, rf, factors):
    x = np.array(factors[['Rm-Rf','SMB','HML','UMD']])
    x = sm.add_constant(x)
    y = np.array(ind)-np.array(rf)
    # alphas
    fama_alpha = []
    # betas
    fama_mkt = []
    fama_smb = []
    fama_hml = []
    fama_umd = []
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        fama_alpha = np.append(fama_alpha, results.params[0])
        fama_mkt = np.append(fama_mkt, results.params[1])
        fama_smb = np.append(fama_smb, results.params[2])
        fama_hml = np.append(fama_hml, results.params[3])
        fama_umd = np.append(fama_umd, results.params[3])
    return fama_alpha,fama_mkt,fama_smb,fama_hml, fama_umd

carh_alpha,carh_mkt,carh_smb,carh_hml,carh_umd = carhart_4factor(Data_Ind, Data_Rf, Data_Factors)

def fama_3factor(ind, rf, factors):
    x = np.array(factors[['Rm-Rf','SMB','HML']])
    x = sm.add_constant(x)
    y = np.array(ind)-np.array(rf)
    # alphas
    fama_alpha = []
    # betas
    fama_mkt = []
    fama_smb = []
    fama_hml = []
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        fama_alpha = np.append(fama_alpha, results.params[0])
        fama_mkt = np.append(fama_mkt, results.params[1])
        fama_smb = np.append(fama_smb, results.params[2])
        fama_hml = np.append(fama_hml, results.params[3])
    
    return fama_alpha,fama_mkt,fama_smb,fama_hml

fama_alpha,fama_mkt,fama_smb,fama_hml = fama_3factor(Data_Ind, Data_Rf, Data_Factors)

# concat the alphas and ratios

comb_alphas = pd.DataFrame([Sharpe,Sortino,Treynor,Jensen,carh_alpha,fama_alpha],
                           index = ['Sharpe','Sortino','Treynor','Jensen','Cahart_alpha','Fama_alpha'],
                           columns = Ind_cols)

comb_alphas = comb_alphas.transpose()
#### plot graph  ##########################################################################################
xloc = np.arange(10)  # the x locations for the groups
    
# Sharpe
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Sharpe'], color = 'k')
plt.title('Sharpe', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)

# Sortino
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Sortino'], color = 'k')
plt.title('Sortino', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)

# Treynor
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Treynor'], color = 'k')
plt.title('Treynor', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)

# Jensen
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Jensen'], color = 'k')
plt.title('Jensen', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)
plt.ylim(-0.5,0.5,10)
# Cahart_alpha
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Cahart_alpha'], color = 'k')
plt.title('Cahart_alpha', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)
plt.ylim(-0.5,0.5,10)
# Fama_alpha
plt.figure(figsize=(8 , 6))
plt.bar(xloc,comb_alphas['Fama_alpha'], color = 'k')
plt.title('Fama_alpha', fontsize = 30, color = 'k')
plt.xticks(xloc, Ind_cols)
plt.ylim(-0.5,0.5,10)