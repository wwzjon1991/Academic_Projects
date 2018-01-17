# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:16:03 2017

@author: Jon Wee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# from excel
industrial_portfolios = 'Industry_Portfolios.xlsx'
Market_Returns = 'Market_Returns.xlsx'
Risk_Factors = 'Risk_Factors.xlsx'

Ind_cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops','Hlth', 'Utils', 'Other']

# Industrial Portfolios
Data_Ind = pd.read_excel(industrial_portfolios)
Data_Ind = Data_Ind.drop('Date', 1)

# Market
Data_Mkt = pd.read_excel(Market_Returns)
Data_Mkt = Data_Mkt.drop('Date', 1)

# Combine Data 
Data_assets = pd.concat([Data_Ind, Data_Mkt], axis=1)

# Rf
Factors = pd.read_excel(Risk_Factors)
Data_Rf = pd.DataFrame(Factors['Rf'])


"""
# Regression one factor if Rf varies month by month
def regression(ind, mkt, rf):
    alpha=[]
    beta = []
    x = np.array(mkt)-np.array(rf)
    x = sm.add_constant(x)
    y = np.array(ind)-np.array(rf)
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        alpha = np.append(alpha, results.params[0])
        beta = np.append(beta, results.params[1])
    return alpha, beta

alpha, beta = regression(Data_Ind, Data_Mkt, Data_Rf)
"""

# Regression one factor where Rf is fixed
def regression(ind, mkt, rf):
    alpha=[]
    beta = []
    x = np.array(mkt)-rf
    x = sm.add_constant(x)
    y = np.array(ind)-rf
    for i in range(10):
        results = sm.OLS(y[:,i],x).fit()
        alpha = np.append(alpha, results.params[0])
        beta = np.append(beta, results.params[1])
    return alpha, beta

alpha, beta = regression(Data_Ind, Data_Mkt, 0.13)

# Mean Returns
Ret_asset = Data_assets.mean()
beta = np.append(beta, 1)

def SML(ret, beta):
    x = beta
    x = sm.add_constant(x)
    y = np.array(ret)
    results = sm.OLS(y,x).fit()
    intercept = results.params[0]
    slope = results.params[1]
    return intercept, slope

intercept, slope = SML(Ret_asset, beta)


def Jensen(ind, rf, mkt):
    
    ind = np.array(ind)
    mkt = np.array(mkt)
    x = mkt-rf
    x = sm.add_constant(x)
    y = ind-rf
    
    alpha = []
    
    for i in range(5):
        results = sm.OLS(y[:,i],x).fit()
        alpha = np.append(alpha, results.params[0])
    return alpha

Jensen = Jensen(Data_Ind, 0.13, Data_Mkt)
print(Jensen)
# Sharpe or Information ratio
def Sharpe(ind, rf):
    excess_ret = np.array(ind)-np.array(rf)
    sharpe = []
    for i in range(10):
        x = excess_ret[:,i].mean()/excess_ret[:,i].std(ddof=1)
        sharpe = np.append(sharpe,x)
    return sharpe

Sharpe = Sharpe(Data_Ind, Data_Rf)
print(Sharpe)
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
print(Sortino)
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
print(Treynor)
# Range of the SML line
Bi = np.linspace( 0 , 2, 100)

######  PLOT GRAPH #############################################################################
plt.figure(figsize=(16 , 10))

# Regress line - SML
# plot(x, y=bi*slope+alpha)
plt.plot(Bi, Bi*slope+intercept, 'k', label = 'Security Market Line', linewidth = 4 )

# Plot Industrial portfolio frontier
plt.plot(beta, Ret_asset, 'ro',  label = 'Industrial Portfolios',  markersize = 15.0 )
plt.plot(beta[10],Ret_asset['Market'], 'bo', label = 'Market Portfolio',  markersize = 15.0 )

# Graph labelling
plt.grid(True)
plt.xlim(0,2,0.25)
plt.ylim(0,2,0.25)
plt.xlabel('Beta', fontsize = 20)
plt.ylabel('Mean return', fontsize = 20)
plt.title('Project HWA 2', fontsize = 30)
plt.legend(fontsize = 18)

plt.show()    