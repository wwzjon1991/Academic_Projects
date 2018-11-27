# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 17:01:36 2018

@author: Jon Wee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr 
from scipy.optimize import minimize
#from datetime import datetime


#%% CSV file, Skiprows
files = {'6_Portfolios_ME_BP_2x3.csv':15,
         '6_Portfolios_ME_CFP_2x3.csv': 19,
         '6_Portfolios_ME_DP_2x3.csv': 19,
         '6_Portfolios_ME_INV_2x3.csv': 16,
         '6_Portfolios_ME_OP_2x3.csv': 16,
         '6_Portfolios_ME_Prior_1_0.csv': 11,
         '6_Portfolios_ME_Prior_12_2.csv': 11 }

# Import excel function with SMB portfolio return | HML portfolio return
def importer(Filename, SkipRows):
    df_excel = pd.read_csv(Filename, skiprows = SkipRows)
    
    df_excel.dropna(inplace = True)
    df_excel.reset_index( drop = True, inplace = True)
    df_excel.columns = ["Date","Small_Lo","Small_Med","Small_Hi","Big_Lo","Big_Med","Big_Hi"]
    
    startdate = df_excel.iloc[:, 0].astype(str).str[0:4]
    startdate = startdate[(startdate == startdate[0]) ]
    startdate = startdate.reset_index()
    
    indexget = []
    
    for i in startdate.index:
        if len(indexget)<2:
            if startdate.iloc[i,0] == 0:
                indexget += [startdate.iloc[i,0]]
            elif startdate.iloc[i,0] > startdate.iloc[i-1,0]+12:
                indexget +=[startdate.iloc[i,0]]
    
    df_value = pd.DataFrame(df_excel.iloc[indexget[0]:indexget[1] , :])
    df_value.index = pd.to_datetime(df_value.loc[:,'Date'] , format='%Y%m')
    
    df_equal = df_excel.iloc[indexget[1]:indexget[1]+indexget[1] , :]
    df_equal.index = pd.to_datetime(df_equal.loc[:,'Date'] , format='%Y%m')
    
    df_value,df_equal = df_value.iloc[:, 1:], df_equal.iloc[:, 1:]
    df_value, df_equal = df_value.astype('float64') , df_equal.astype('float64')
     
    df_value = df_value.assign(SmallCap=np.nan,LargeCap=np.nan,SmallCapCumRet=np.nan,LargeCapCumRet=np.nan,SLalpha=np.nan,
                               LEalpha=np.nan, SLbeta=np.nan, LEbeta=np.nan) 
    df_equal = df_equal.assign(SmallCap=np.nan,LargeCap=np.nan,SmallCapCumRet=np.nan,LargeCapCumRet=np.nan,SLalpha=np.nan,
                               LEalpha=np.nan, SLbeta=np.nan, LEbeta=np.nan) 
    return df_value, df_equal

#%% Part I - Monthly only
    
# Fama-French
df_FF = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3)
df_FF.columns = ["Date","MKT-RF","SMB","HML", "RF"] 
df_FF = df_FF.iloc[:1101, :]
df_FF.index = pd.to_datetime(df_FF.loc[:,'Date'] , format='%Y%m')
df_FF = df_FF.iloc[:, 1:]

#df_FF['Ones'] = np.ones(df_FF.shape[0]) 
df_FF = df_FF.astype('float64')
#df_FF = df_FF[["Ones","MKT-RF","SMB","HML", "RF"]]

# Book to Price
df_BP_value, df_BP_equal = importer(list(files.keys())[0], list(files.values())[0])

# Cashflow to Price
df_CP_value, df_CP_equal = importer(list(files.keys())[1], list(files.values())[1])

# Dividend Yield
df_DY_value, df_DY_equal = importer(list(files.keys())[2], list(files.values())[2])

# Investment
df_INV_value, df_INV_equal = importer(list(files.keys())[3], list(files.values())[3])

# Profitability
df_PFT_value, df_PFT_equal = importer(list(files.keys())[4], list(files.values())[4])

# Price Momentum # 1 month 
df_PM1M_value, df_PM1M_equal = importer(list(files.keys())[5], list(files.values())[5])
# reduce size to match FF mkt
df_PM1M_value, df_PM1M_equal = df_PM1M_value[-len(df_FF):], df_PM1M_equal[-len(df_FF):]

# Price Momentum # 12 month 
df_PM12M_value, df_PM12M_equal = importer(list(files.keys())[6], list(files.values())[6])

#%% 
zipp_files = [df_BP_value,df_BP_equal,
              df_CP_value, df_CP_equal,
              df_DY_value, df_DY_equal,
              df_PFT_value, df_PFT_equal,
              df_PM12M_value, df_PM12M_equal,
              df_INV_value, df_INV_equal,
              df_PM1M_value, df_PM1M_equal]
# Reshape all the data to match
minshape = -min(x.shape[0] for x in zipp_files)

# reduce size Fama-French
df_FF = df_FF.iloc[minshape:, :]
df_BP_value,df_BP_equal = df_BP_value.iloc[minshape:, :],df_BP_equal.iloc[minshape:, :]
df_CP_value, df_CP_equal = df_CP_value.iloc[minshape:, :], df_CP_equal.iloc[minshape:, :]
df_DY_value, df_DY_equal = df_DY_value.iloc[minshape:, :], df_DY_equal.iloc[minshape:, :]
df_PFT_value, df_PFT_equal = df_PFT_value.iloc[minshape:, :], df_PFT_equal.iloc[minshape:, :] 
df_PM12M_value, df_PM12M_equal = df_PM12M_value.iloc[minshape:, :], df_PM12M_equal.iloc[minshape:, :]
df_INV_value, df_INV_equal = df_INV_value.iloc[minshape:, :], df_INV_equal.iloc[minshape:, :]
df_PM1M_value, df_PM1M_equal = df_PM1M_value.iloc[minshape:, :], df_PM1M_equal.iloc[minshape:, :]

# Excluded Price Momentum : Cuz dono how to do the returns
zipp_files = [df_BP_value,df_BP_equal,
              df_CP_value, df_CP_equal,
              df_DY_value, df_DY_equal,
              df_PFT_value, df_PFT_equal,
              df_PM12M_value, df_PM12M_equal,
              df_INV_value, df_INV_equal,
              df_PM1M_value, df_PM1M_equal] 

zipp_title = ['Book to Price value','Book to Price equal',
              'Cash to Price value', 'Cash to Price equal',
              'Dividend Yield value', 'Dividend Yield equal',
              'Profitablilty value', 'Profitability equal',
              'Pior 12-1 months value', 'Pior 12-1 months equal',
              'Investment value', 'Investment equal',
              'Pior 1 month value', 'Pior 1 month equal'] 


#%% Cumulative Returns

def SmallBigGraph(Small, Big, title):
        
    plt.plot(Small.index,Small,'r-.',label='Small Cap')
    plt.plot(Small.index,Big,'b-.',label='Large Cap')
    
    plt.title(title, color = 'k')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

for id,factor in enumerate(zipp_files):
    # Small Cap | Large Cap Portfoilos returns
    if id<10:
        factor['SmallCap'] = factor["Small_Hi"] - factor["Small_Lo"] 
    
        factor['LargeCap'] = factor["Big_Hi"] - factor["Big_Lo"] 
        factor['SmallCapCumRet'], factor['LargeCapCumRet'] = \
                                    factor['SmallCap'].cumsum(), factor['LargeCap'].cumsum()
    else:
        factor['SmallCap'] =  factor["Small_Lo"] - factor["Small_Hi"]
    
        factor['LargeCap'] = factor["Big_Lo"] -  factor["Big_Hi"]
        factor['SmallCapCumRet'], factor['LargeCapCumRet'] = \
                                    factor['SmallCap'].cumsum(), factor['LargeCap'].cumsum()

    if (id)%2 == 0:
        plt.figure(figsize = (12, 12))
        i = 0
        
    plt.subplot(2, 1, i+1)
    SmallBigGraph(factor['SmallCapCumRet'], factor['LargeCapCumRet'], zipp_title[id])    
    i+=1

#%% Regression
def mktbeta(SMB, HML, title):
    
    SMB = SMB.dropna()
    HML = HML.dropna()
    
    plt.plot(SMB.index,SMB,'r-.',label='Small Cap')
    plt.plot(HML.index,HML,'b-.',label='Large Cap')
    
    plt.title(title, color = 'k')
    plt.xlabel('Date')
    plt.ylabel('Market Beta')
    plt.legend()
    plt.show()

def Fama_French(FF_Factors, PortfolioReturns):
    
    x = FF_Factors["MKT-RF"].values
    x = sm.add_constant(x)
    y = PortfolioReturns.values - FF_Factors["RF"].values
    results = sm.OLS(y,x).fit()

    return results.params[0],results.params[1]
    


Shape = len(df_FF)
for id, factor in enumerate(zipp_files):
   
    for rolls in range(36, Shape+1,1):
        SLalpha, SLbeta = Fama_French(df_FF.iloc[ rolls-36:rolls , :], 
                                      factor['SmallCap'].iloc[rolls-36:rolls])
        LEalpha, LEbeta = Fama_French(df_FF.iloc[ rolls-36:rolls , :], 
                                      factor['LargeCap'].iloc[rolls-36:rolls])
        
        factor['SLalpha'].iloc[rolls-1] = SLalpha
        factor['LEalpha'].iloc[rolls-1] = LEalpha
        
        factor['SLbeta'].iloc[rolls-1] = SLbeta
        factor['LEbeta'].iloc[rolls-1] = LEbeta
    
    if (id)%2 == 0:
        
        plt.figure(figsize = (12, 12))
        i = 0
        
    plt.subplot(2, 1, i+1)
    mktbeta(factor.iloc[: , -2], factor.iloc[: , -1], zipp_title[id])    
    i+=1


#%% Beta Neutral
  
reg_BP_equal = df_BP_equal.iloc[:, :6]
reg_BP_equal = reg_BP_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_CP_equal = df_CP_equal.iloc[:, :6]
reg_CP_equal = reg_CP_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_DY_equal = df_DY_equal.iloc[:, :6]
reg_DY_equal = reg_DY_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_INV_equal = df_INV_equal.iloc[:, :6]
reg_INV_equal = reg_INV_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_PFT_equal = df_PFT_equal.iloc[:, :6]
reg_PFT_equal = reg_PFT_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_Pior1_equal = df_PM1M_equal.iloc[:, :6]
reg_Pior1_equal = reg_Pior1_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

reg_Pior12_1_equal = df_PM12M_equal.iloc[:, :6]
reg_Pior12_1_equal = reg_Pior12_1_equal.assign(AlphaSLo=np.nan,AlphaSHi=np.nan,AlphaBLo=np.nan,AlphaBHi=np.nan,
                                   BetaSLo=np.nan, BetaSHi=np.nan, BetaBLo=np.nan, BetaBHi=np.nan,
                                   WSmall_Lo=np.nan, WSmall_Hi=np.nan, WBig_Lo=np.nan, WBig_Hi=np.nan,
                                   Weights = np.nan, Pbeta=np.nan ,BNrets = np.nan)

zip_reg = [reg_BP_equal, reg_CP_equal, reg_DY_equal, reg_PFT_equal, reg_Pior12_1_equal, reg_INV_equal, reg_Pior1_equal]

#%%  REGRESSION
for id, factor in enumerate(zip_reg):
    
    for rolls in range(36, Shape+1,1):
        
        AlphaSLo, BetaSLo = Fama_French(df_FF.iloc[ rolls-36:rolls , :], factor['Small_Lo'].iloc[rolls-36:rolls])
        AlphaSHi, BetaSHi = Fama_French(df_FF.iloc[ rolls-36:rolls , :], factor['Small_Hi'].iloc[rolls-36:rolls])
        
        AlphaBLo, BetaBLo = Fama_French(df_FF.iloc[ rolls-36:rolls , :], factor['Big_Lo'].iloc[rolls-36:rolls])
        AlphaBHi, BetaBHi = Fama_French(df_FF.iloc[ rolls-36:rolls , :], factor['Big_Hi'].iloc[rolls-36:rolls])
        
        factor['AlphaSLo'].iloc[rolls-1] = AlphaSLo
        factor['AlphaSHi'].iloc[rolls-1] = AlphaSHi
        factor['AlphaBLo'].iloc[rolls-1] = AlphaBLo
        factor['AlphaBHi'].iloc[rolls-1] = AlphaBHi

        factor['BetaSLo'].iloc[rolls-1] = BetaSLo
        factor['BetaSHi'].iloc[rolls-1] = BetaSHi
        factor['BetaBLo'].iloc[rolls-1] = BetaBLo
        factor['BetaBHi'].iloc[rolls-1] = BetaBHi
        
reg_BP_equal = reg_BP_equal.loc['1966-05-01 00:00:00':, :]
reg_CP_equal = reg_CP_equal.loc['1966-05-01 00:00:00':, :]
reg_DY_equal = reg_DY_equal.loc['1966-05-01 00:00:00':, :]
reg_PFT_equal = reg_PFT_equal.loc['1966-05-01 00:00:00':, :]
reg_Pior12_1_equal = reg_Pior12_1_equal.loc['1966-05-01 00:00:00':, :]
reg_INV_equal = reg_INV_equal.loc['1966-05-01 00:00:00':, :]
reg_Pior1_equal = reg_Pior1_equal.loc['1966-05-01 00:00:00':, :] 

zip_reg = [reg_BP_equal, reg_CP_equal, reg_DY_equal,
           reg_PFT_equal, reg_Pior12_1_equal, reg_INV_equal, reg_Pior1_equal]
    
#%% Dynamic weighting for beta neutral

#maximise returns
def rets(x):
    Port = x[0]*ret[0]+ x[1]*ret[1]+ x[2]*ret[2]+ x[3]*ret[3]
    return -Port

def constraint1(x):
    # '1' stands for the inital investment amt
    # cash is the short sell borrowing amt: Risk-Free rate
    # long amt + short amt = inital investment
    if zipID <5:
        return 1*(x[1]+x[3])+cash*(x[0]+x[2])-1 
    else:
        return 1*(x[0]+x[2])+cash*(x[1]+x[3])-1 
# DOLLAR NEUTRAL
def constraint2(x):
    return x[1]+x[3]+(x[0]+x[2])

# BETA NEUTRAL
def constraint3(x):
    return x[1]*beta[1]+x[3]*beta[3]+(x[0]*beta[0]+x[2]*beta[2])

x0 = [-0.5, 0.5, -1.5, 1.5] #guess
x1 = [0.5, -0.5, 1.5, -1.5] #guess
bnds1 = ((-50,0),(0,None),(-50,0),(0,None)) # bounds(min, max)
bnds2 = ((0,None),(-50,0),(0,None),(-50,0)) # bounds(min, max)
Shape = len(reg_BP_equal) # Shape of factor

for zipID, factor in enumerate(zip_reg):
    for id in range(Shape):
        cash = df_FF['RF'][34+id]
        
        # returns or alpha??
        ret  = [factor['Small_Lo'][id], factor['Small_Hi'][id], \
                factor['Big_Lo'][id], factor['Big_Hi'][id]]
        
        beta = [factor['BetaSLo'][id], factor['BetaSHi'][id], 
                factor['BetaBLo'][id], factor['BetaBHi'][id]]
        
        con1 = {'type': 'eq', 'fun': constraint1} # eq indicates equality
        con2 = {'type': 'eq', 'fun': constraint2}
        con3 = {'type': 'eq', 'fun': constraint3}
        cons = [con1, con2, con3] 
    
        if zipID <5:
            sol = minimize(rets,x0,method='SLSQP',bounds=bnds1, constraints=cons)
        else:
            sol = minimize(rets,x1,method='SLSQP',bounds=bnds2, constraints=cons)
    #    sol = minimize(constraint3,x0,method='SLSQP',bounds=bnds, constraints=con2)
        factor['WSmall_Lo'][id] = sol.x[0]   
        factor['WSmall_Hi'][id] = sol.x[1]   
        factor['WBig_Lo'][id]   = sol.x[2]   
        factor['WBig_Hi'][id]   = sol.x[3]   
        
        factor['BNrets'][id] = sol.x[0]*ret[0] + sol.x[1]*ret[1] \
                                   + sol.x[2]*ret[2] + sol.x[3]*ret[3]
        # check if is beta neutral                           
        factor['Pbeta'][id] = sol.x[0]*beta[0] + sol.x[1]*beta[1] \
                                   + sol.x[2]*beta[2] + sol.x[3]*beta[3]
        # check if is dollar neutral                           
        factor['Weights'][id] = np.sum(sol.x)
        
    # plot    
    cumret = factor.dropna()
    equal = zipp_files[zipID*2+1].dropna().iloc[:, 6:8]    
    plt.figure(figsize = (12, 12))
    plt.plot(equal.index,equal.iloc[:,0].cumsum(),'r-.',label='Small Cap')
    plt.plot(equal.index,equal.iloc[:,1].cumsum(),'b-.',label='Large Cap')
    plt.plot(cumret.index,cumret.iloc[:,-1].cumsum(),'g-.',label='Beta Neutral')
    plt.title(zipp_title[zipID*2+1])
    plt.legend()
    plt.show()

#%% Part II 
#Correlation Matrix
"""
zipp_files = [df_BP_value,df_BP_equal,
              df_CP_value, df_CP_equal,
              df_DY_value, df_DY_equal,
              df_PFT_value, df_PFT_equal,
              df_PM12M_value, df_PM12M_equal,
              df_INV_value, df_INV_equal,
              df_PM1M_value, df_PM1M_equal] 

zipp_title = ['Book to Price value','Book to Price equal',
              'Cash to Price value', 'Cash to Price equal',
              'Dividend Yield value', 'Dividend Yield equal',
              'Profitablilty value', 'Profitability equal',
              'Pior 12-1 months value', 'Pior 12-1 months equal',
              'Investment value', 'Investment equal',
              'Pior 1 month value', 'Pior 1 month equal'] 
"""
matrix_ind = ['BP' , 'CP' ,'DY','Pft', 'Pior_12-1', 'Inv','Pior_1']
df_corr_data = pd.DataFrame(columns = matrix_ind)

for col,factor in enumerate(range(1, len(zipp_files),2)): # get equal weight factors only
    small , big = zipp_files[factor].iloc[:, 6], zipp_files[factor].iloc[:, 7]
    df_corr_data.iloc[:,col] = small*0.5 + big*0.5
    SmallLo , SmallHi = zipp_files[factor].iloc[:, 0], zipp_files[factor].iloc[:, 2]
    BigLo , BigHi = zipp_files[factor].iloc[:, 3], zipp_files[factor].iloc[:, 5]
    if col < 5: 
        df_corr_data.iloc[:,col] = (SmallHi + BigHi )*0.5 - (SmallLo+BigLo)*0.5
    else: 
        df_corr_data.iloc[:,col] = -(SmallHi + BigHi )*0.5 + (SmallLo+BigLo)*0.5

# 5 year look back for equal weighted
df_cov_matrix  = df_corr_data.cov()    
df_corr_matrix = df_corr_data.corr(method='pearson')

#%%
# min var optimization
def calculate_portfolio_var(w,V):
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

# Dollar neutral
def cons1(w):
    return np.sum(w)

# Port returns must be greater than mkt returns
def cons2(w):
    w = np.matrix(w)
    port = np.sum(w*R)
    return port-benchmark

Shape = len(df_corr_data)

df_minvar = pd.DataFrame(index =df_corr_data.index, columns = matrix_ind, dtype = np.float32 )
df_minvar = df_minvar.assign(PortRet=np.nan, var=np.nan, EqRiskRet = np.nan) 

# Lower Boundary for weights = 0.1428
lb = -np.float16((100/7)/100)
bnds1 = ((lb,None),(lb,None),(lb,None),(lb,None),(lb,None),(lb,None),(lb,None)) # bounds(min, max)

for id in range(59, Shape):
    
    # covariance matrix
    df_cov_matrix  = df_corr_data.iloc[id-60:id , :].cov() 
    
    # Market Return
    benchmark = df_FF.iloc[id, 0]
    
    V = np.matrix(df_cov_matrix)
    R = np.matrix(df_corr_data.iloc[id, :]).T
    rf = df_FF.iloc[id, -1]
    
    w0= [0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    
    # unconstrained portfolio (only sum(w) = 1 )
    c1 = ({'type': 'eq', 'fun': cons1})
    c2 = ({'type': 'ineq', 'fun': cons2})
    cons = [c1, c2]
    res= minimize(calculate_portfolio_var, w0, args=V, method='SLSQP',bounds=bnds1,constraints=cons)
    w_g = res.x
    port_g = w_g*R
    var_g = np.dot(w_g*V,w_g)
    
    EqriskW = [-lb,-lb,-lb,-lb,-lb,-lb,-lb]
    
    df_minvar.iloc[id, :7 ] = w_g
    df_minvar.iloc[id,  7: ] = port_g,var_g, EqriskW*R 

    
    
df_minvar1 = df_minvar.dropna()   
plt.figure(figsize = (12, 12))
plt.plot(df_minvar1.index, df_minvar1.iloc[:,7].cumsum(),'r-.',label='Dynamic Weighting')
plt.plot(df_minvar1.index,df_minvar1.iloc[:,9].cumsum(),'b-.',label='Equal-Risk Weighting')
#plt.plot(cumret.index,cumret.iloc[:,-1].cumsum(),'g-.',label='Beta Neutral')
plt.title("Dynamic vs Static equally weighted")
plt.legend()
plt.show()



#%% Part III





























