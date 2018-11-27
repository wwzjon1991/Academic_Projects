# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 17:01:36 2018

@author: Jon Wee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.stats import pearsonr 
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime

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
     
    df_value = df_value.assign(SmallCapRet=np.nan,BigCapRet=np.nan,SmallCapCumRet=np.nan,BigCapCumRet=np.nan,SLalpha=np.nan,
                               LEalpha=np.nan, SLbeta=np.nan, LEbeta=np.nan) 
    df_equal = df_equal.assign(SmallCapRet=np.nan,BigCapRet=np.nan,SmallCapCumRet=np.nan,BigCapCumRet=np.nan,SLalpha=np.nan,
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
              'Prior 12-1 months value', 'Prior 12-1 months equal',
              'Investment value', 'Investment equal',
              'Prior 1 month value', 'Prior 1 month equal'] 


#%% Cumulative Returns

def SmallBigGraph(Small, Big, title):
        
    plt.plot(Small.index,Small,'r-.',label='Small Cap')
    plt.plot(Small.index,Big,'b-.',label='Big Cap')
    
    plt.title(title, color = 'k')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    x = Small.index
    plt.xlim(x[0], x[-1])
    plt.show()
    plt.savefig('Part1A'+zipp_title[id]+'CumulativeRet.png')

for id,factor in enumerate(zipp_files):
    # Small Cap | Large Cap Portfoilos returns
    if id<10:
        factor['SmallCapRet'] = factor["Small_Hi"] - factor["Small_Lo"] 
        factor['BigCapRet'] = factor["Big_Hi"] - factor["Big_Lo"] 
        # Cumulative Returns
        factor['SmallCapCumRet'], factor['BigCapCumRet'] = \
                                    factor['SmallCapRet'].cumsum(), factor['BigCapRet'].cumsum()
    else:
        factor['SmallCapRet'] =  factor["Small_Lo"] - factor["Small_Hi"]
        factor['BigCapRet'] = factor["Big_Lo"] -  factor["Big_Hi"]
        # Cumulative Returns
        factor['SmallCapCumRet'], factor['BigCapCumRet'] = \
                                    factor['SmallCapRet'].cumsum(), factor['BigCapRet'].cumsum()

    if (id)%2 == 0:
        plt.figure(figsize = (15, 10))
        i = 0
        
    plt.subplot(2, 1, i+1)
    SmallBigGraph(factor['SmallCapCumRet'], factor['BigCapCumRet'], zipp_title[id])    
    i+=1

#%% Ratios for Normal Cumulative REturns
def MaxDrawndown(xs):
    i = np.argmax(np.maximum.accumulate(xs)-xs)
    j = np.argmax(xs[:i])
    return xs[i]-xs[j]    
    
Pt1Ratiosind = ['Book to Price',
                'Cash to Price',
                'Dividend Yield',
                'Profitablilty',
                'Prior 12-1 months',
                'Investment',
                'Prior 1 month']

Pt1Ratioscol = ['Small-Volatility', 'Small-Mean-Return', 'Small-Sharpe Ratio','Small-MaxDrawdown', \
                'Big-Volatility', 'Big-Mean-Return', 'Big-Sharpe Ratio','Big-MaxDrawdown']

VaPt1ARatios = pd.DataFrame(index = Pt1Ratiosind , columns = Pt1Ratioscol, dtype='float')
EqPt1ARatios = pd.DataFrame(index = Pt1Ratiosind , columns = Pt1Ratioscol, dtype='float')

for port in range(len(Pt1Ratiosind)):
    # Volatility
    VaPt1ARatios.iloc[port , 0] =  zipp_files[port].iloc[: , 6].std()
    VaPt1ARatios.iloc[port , 4] =  zipp_files[port].iloc[: , 7].std()
    
    EqPt1ARatios.iloc[port , 0] =  zipp_files[port+1].iloc[: , 6].std()
    EqPt1ARatios.iloc[port , 4] =  zipp_files[port+1].iloc[: , 7].std()
    # Mean Return
    VaPt1ARatios.iloc[port , 1] =  np.mean(zipp_files[port].iloc[: , 6])
    VaPt1ARatios.iloc[port , 5] =  np.mean(zipp_files[port].iloc[: , 7])
    
    EqPt1ARatios.iloc[port , 1] =  np.mean(zipp_files[port+1].iloc[: , 6])
    EqPt1ARatios.iloc[port , 5] =  np.mean(zipp_files[port+1].iloc[: , 7])    
    # Sharpe Ratio
    VaPt1ARatios.iloc[port , 2] =  np.mean(zipp_files[port].iloc[: , 6]-df_FF['RF']) /zipp_files[port].iloc[: , 6].std()
    VaPt1ARatios.iloc[port , 6] =  np.mean(zipp_files[port].iloc[: , 7]-df_FF['RF']) /zipp_files[port].iloc[: , 7].std()
    
    EqPt1ARatios.iloc[port , 2] =  np.mean(zipp_files[port+1].iloc[: , 6]-df_FF['RF']) /zipp_files[port+1].iloc[: , 6].std()
    EqPt1ARatios.iloc[port , 6] =  np.mean(zipp_files[port+1].iloc[: , 7]-df_FF['RF']) /zipp_files[port+1].iloc[: , 7].std()    
    # Max Drawdown
    VaPt1ARatios.iloc[port , 3] =  MaxDrawndown(zipp_files[port].iloc[: , 8])
    VaPt1ARatios.iloc[port , 7] =  MaxDrawndown(zipp_files[port].iloc[: , 9])
    
    EqPt1ARatios.iloc[port , 3] =  MaxDrawndown(zipp_files[port+1].iloc[: , 8])
    EqPt1ARatios.iloc[port , 7] =  MaxDrawndown(zipp_files[port+1].iloc[: , 9])
    
    
print(VaPt1ARatios)
print(EqPt1ARatios)
    
#%% Regression
def mktbeta(SMB, HML, title):
    
    SMB = SMB.dropna()
    HML = HML.dropna()
    
    plt.plot(SMB.index,SMB,'r-.',label='Small Cap')
    plt.plot(HML.index,HML,'b-.',label='Big Cap')
    
    plt.title(title, color = 'k')
    plt.xlabel('Date')
    plt.ylabel('Market Beta')
    plt.legend()
    x = SMB.index
    plt.xlim(x[0], x[-1])
    plt.show()
    plt.savefig('Part1B'+zipp_title[id]+'MKTBeta.png')


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
                                      factor['SmallCapRet'].iloc[rolls-36:rolls])
        LEalpha, LEbeta = Fama_French(df_FF.iloc[ rolls-36:rolls , :], 
                                      factor['BigCapRet'].iloc[rolls-36:rolls])
        
        factor['SLalpha'].iloc[rolls-1] = SLalpha
        factor['LEalpha'].iloc[rolls-1] = LEalpha
        
        factor['SLbeta'].iloc[rolls-1] = SLbeta
        factor['LEbeta'].iloc[rolls-1] = LEbeta
    
    if (id)%2 == 0:
        
        plt.figure(figsize = (15, 10))
        i = 0
        
    plt.subplot(2, 1, i+1)
    mktbeta(factor.iloc[: , -2], factor.iloc[: , -1], zipp_title[id])    
    i+=1
#%% Ratio 
Pt1Ratioscol = ['Value Small Cap', 'Value Big Cap', 'Equal Small Cap', 'Equal Big Cap']    
Pt1Ratiosind = ['Book to Price',
                'Cash to Price',
                'Dividend Yield',
                'Profitablilty',
                'Prior 12-1 months',
                'Investment',
                'Prior 1 month'] 
Pt1BetaMean = pd.DataFrame(index =Pt1Ratiosind , columns = Pt1Ratioscol, dtype = 'float')
Pt1BetaStd = pd.DataFrame(index =Pt1Ratiosind , columns = Pt1Ratioscol, dtype = 'float')
zipp_files = [df_BP_value,df_BP_equal,
              df_CP_value, df_CP_equal,
              df_DY_value, df_DY_equal,
              df_PFT_value, df_PFT_equal,
              df_PM12M_value, df_PM12M_equal,   
              df_INV_value, df_INV_equal,
              df_PM1M_value, df_PM1M_equal] 
for id,col in enumerate(range(0 ,len(zipp_files),2)):   
    print(col)
    # mean 
    Pt1BetaMean.iloc[id, 0] = zipp_files[col].iloc[: , -2].dropna().mean()
    Pt1BetaMean.iloc[id, 1] = zipp_files[col].iloc[: , -1].dropna().mean()
    Pt1BetaMean.iloc[id, 2] = zipp_files[col+1].iloc[: , -2].dropna().mean()
    Pt1BetaMean.iloc[id, 3] = zipp_files[col+1].iloc[: , -1].dropna().mean()
    
    Pt1BetaStd.iloc[id, 0] = zipp_files[col].iloc[: , -2].dropna().std()
    Pt1BetaStd.iloc[id, 1] = zipp_files[col].iloc[: , -1].dropna().std()
    Pt1BetaStd.iloc[id, 2] = zipp_files[col+1].iloc[: , -2].dropna().std()
    Pt1BetaStd.iloc[id, 3] = zipp_files[col+1].iloc[: , -1].dropna().std()    

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
#    Port = x[0]*ret[0]+ x[1]*ret[1]+ x[2]*ret[2]+ x[3]*ret[3]
    Port = x[0]*alp[0]+ x[1]*alp[1]+ x[2]*alp[2]+ x[3]*alp[3]
#    Port = -(x[0]*var[0]+ x[1]*var[1]+ x[2]*var[2]+ x[3]*var[3])
    return -Port

def constraint1(x):
    # '1' stands for the inital investment amt
    # 0.1 is the cash budget
    # long amt + short amt = inital investment
    if zipID <5:
        return 1*(x[1]+x[3])+0.1*(x[0]+x[2])-1 
    else:
        return 1*(x[0]+x[2])+0.1*(x[1]+x[3])-1

#DOLLAR NEUTRAL
#def constraint2(x):
#    return x[1]+x[3]+(x[0]+x[2])
   
#def constraint2(x):
#    if zipID<5:
#        return x[1]+x[3]-Z-0.2
#    else:
#        return x[0]+x[2]-Z-0.2
#def constraint3(x):
#    if zipID<5:
#        return x[0]+x[2]+Z+0.2
#    else:
#        return x[1]+x[3]+Z+0.2

# BETA NEUTRAL
def constraint4(x):
    return x[1]*beta[1]+x[3]*beta[3]+(x[0]*beta[0]+x[2]*beta[2])


# Loosen Constraint on Long position by 0.5 weights
Z = 1
bnds1 = ((-Z,0),(0,Z),(-Z,0),(0,Z)) # bounds(min, max)
bnds2 = ((0,Z),(-Z,0),(0,Z),(-Z,0)) # bounds(min, max)

# Initial Guess
G = 0.1
x0 = [-G, G, -G, G] 
x1 = [G, -G, G, -Z] 

Shape = len(reg_BP_equal) # Shape of factor
Beta = ['BP' , 'CP' ,'DY','Pft', 'Prior_12-1', 'Inv','Prior_1']

for zipID, factor in enumerate(zip_reg):
    for id in range(Shape):
        # returns or alpha??
        alp  = [factor['AlphaSLo'][id], factor['AlphaSHi'][id], \
                factor['AlphaBLo'][id], factor['AlphaBHi'][id]]
        
        ret  = [factor['Small_Lo'][id], factor['Small_Hi'][id], \
                factor['Big_Lo'][id], factor['Big_Hi'][id]]
        
        var  = [factor['Small_Lo'][id].std(), factor['Small_Hi'][id].std(), \
                factor['Big_Lo'][id].std(), factor['Big_Hi'][id].std()]
        
        beta = [factor['BetaSLo'][id], factor['BetaSHi'][id], 
                factor['BetaBLo'][id], factor['BetaBHi'][id]]
        
        con1 = {'type': 'eq', 'fun': constraint1} # eq indicates equality
        con2 = {'type': 'eq', 'fun': constraint4}
        cons = [con1,con2] 
    
        if zipID <5:
            sol = minimize(rets,x0,method='SLSQP',bounds=bnds1, constraints=cons)
        else:
            sol = minimize(rets,x1,method='SLSQP',bounds=bnds2, constraints=cons)
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
    plt.figure(figsize = (15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(equal.index,equal.iloc[:,0].cumsum(),'r-.',label='Small Cap')
    plt.plot(equal.index,equal.iloc[:,1].cumsum(),'b-.',label='Large Cap')
    plt.plot(cumret.index,cumret.iloc[:,-1].cumsum(),'g-.',label='Beta Neutral')
    plt.title(zipp_title[zipID*2+1])
    plt.legend()
    x = equal.index
    plt.xlim(x[0], x[-1])
    plt.subplot(2, 1, 2)
    plt.plot(zip_reg[zipID]['Pbeta'].dropna().index,zip_reg[zipID]['Pbeta'].dropna(), color='r', label = Beta[zipID]+' Beta' )    
    x = zip_reg[zipID]['Pbeta'].index
    plt.xlim(x[0], x[-1])
    plt.legend() 
    plt.show()
    plt.savefig('Part1C'+zipp_title[zipID*2+1]+'BetaNeutral.png')

for i in range(7):
    plt.figure(figsize = (15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(zip_reg[i]['Pbeta'].dropna().index,zip_reg[i]['Pbeta'].dropna(), color='r', label = Beta[i]+' Beta' )    
    plt.legend() 
    plt.subplot(2, 1, 2)
    plt.bar(zip_reg[i]['Weights'].dropna().index,zip_reg[i]['Weights'].dropna(), width = 10,color='b', label = Beta[i]+' Weights' )
    plt.legend() 
    plt.show()
    plt.savefig(Beta[i]+' Beta' +'Pt1Beta.png')
#%% Ratios for Beta Neutral Portfolios

zip_reg = [reg_BP_equal, reg_CP_equal, reg_DY_equal,
           reg_PFT_equal, reg_Pior12_1_equal, reg_INV_equal, reg_Pior1_equal]
shappe = len(reg_BP_equal.dropna())
Pt1Ratioscol = ['Mean', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']    
Pt1Ratiosind = ['Book to Price',
                'Cash to Price',
                'Dividend Yield',
                'Profitablilty',
                'Prior 12-1 months',
                'Investment',
                'Prior 1 month'] 
Pt1CRatio = pd.DataFrame(index =Pt1Ratiosind , columns = Pt1Ratioscol, dtype = 'float')

for id,col in enumerate(range(0 ,len(zip_reg))):   
    # mean 
    Pt1CRatio.iloc[id, 0] = zip_reg[col].iloc[: , -1].dropna().mean()
    # volatility
    Pt1CRatio.iloc[id, 1] = zip_reg[col].iloc[: , -1].dropna().std()
    # Sharpe
    Pt1CRatio.iloc[id , 2] =  np.mean(zip_reg[col].iloc[: , -1].dropna()-df_FF.iloc[-shappe:,-1])\
                                /zip_reg[col].iloc[: , -1].dropna().std()    
    # Max Drawdown
    Pt1CRatio.iloc[id , 3] =  MaxDrawndown(zip_reg[col].iloc[: , 1])
    
    
print(Pt1CRatio)
    

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
"""
matrix_ind = ['BP' , 'CP' ,'DY','Pft', 'Prior_12-1', 'Inv','Prior_1']
df_Va = pd.DataFrame(columns = matrix_ind)
df_Eq = pd.DataFrame(columns = matrix_ind)


for col,factor in enumerate(matrix_ind): # get equal weight factors only
    # Value
    VaSmallLo , VaSmallHi = zipp_files[col*2].iloc[:, 0], zipp_files[col*2].iloc[:, 2]
    VaBigLo , VaBigHi = zipp_files[col*2].iloc[:, 3], zipp_files[col*2].iloc[:, 5]
    # Equal
    EqSmallLo , EqSmallHi = zipp_files[col*2+1].iloc[:, 0], zipp_files[col*2+1].iloc[:, 2]
    EqBigLo , EqBigHi = zipp_files[col*2+1].iloc[:, 3], zipp_files[col*2+1].iloc[:, 5]

    if col < 5: 
        df_Va.iloc[:,col] = (VaSmallHi-VaSmallLo)*0.5 + (VaBigHi-VaBigLo)*0.5
        df_Eq.iloc[:,col] = (EqSmallHi-EqSmallLo)*0.5 + (EqBigHi-EqBigLo)*0.5
    else: 
        df_Va.iloc[:,col] = (-VaSmallHi+VaSmallLo)*0.5 + (-VaBigHi+VaBigLo)*0.5
        df_Eq.iloc[:,col] = (-EqSmallHi+EqSmallLo)*0.5 + (-EqBigHi+EqBigLo)*0.5 

                   

# 5 year look back for equal weighted
df_Eqcov, df_Vacov  = df_Eq.cov() , df_Va.cov()     
df_Eqcorr, df_Vacorr = df_Eq.corr(method='pearson'), df_Va.corr(method='pearson')


#print("Equal-Weight Portfolio Correlations:")
#print(df_Eqcorr)
#print("Value-Weight Portfolio Correlations:")
#print(df_Vacorr)
# Correlation Matrix with Mkt
df_VaC = df_Va.iloc[:, :7]
df_VaC['Mkt'] = df_FF['MKT-RF']
df_EqC = df_Eq.iloc[:, :7]
df_EqC['Mkt'] = df_FF['MKT-RF']
df_EqCcorr, df_VaCcorr = df_EqC.corr(method='pearson'), df_VaC.corr(method='pearson')
print("Equal-Weight Portfolio Correlations:")
print(df_EqCcorr)
print("Value-Weight Portfolio Correlations:")
print(df_VaCcorr)

#%%
# ERC optimization
def calculate_portfolio_var(w,V):
    w = np.matrix(w)
    Pvol = np.sqrt((w*V*w.T)[0,0])# std
    
    Riskcon = np.multiply(V*w.T,w.T)/Pvol
    
    x_t = [1/7]*7
    risk_target = np.asmatrix(np.multiply(Pvol, x_t))
    
    return sum(np.square(Riskcon - risk_target.T))[0,0] # sum of squared error
    
# Dollar neutral
def cons1(w):
    return np.sum(w)-1

df_DWVa = pd.DataFrame(index = df_Va.index ,columns = matrix_ind, dtype = 'float')
#df_SWVa = pd.DataFrame(index = df_Va.index ,columns = matrix_ind)
df_DWEq = pd.DataFrame(index = df_Va.index ,columns = matrix_ind, dtype = 'float')
#df_SWEq = pd.DataFrame(index = df_Va.index ,columns = matrix_ind)

df_Va = df_Va.assign(DW=np.nan, STD=np.nan, SW=np.nan)
df_Eq = df_Eq.assign(DW=np.nan, STD=np.nan, SW=np.nan)

portfolio = [df_Va, df_Eq]
Wportfolio = [df_DWVa, df_DWEq]
port_title = ['Value', 'Equal']

# Lower Boundary for weights = 0.1428
lb = np.float64(1/7)
for num,port in enumerate(portfolio):
    for id in range(60, len(df_Va)):
        # covariance matrix
        df_cov_matrix  = port.iloc[id-60:id , :7].cov() 
                
        V = np.matrix(df_cov_matrix)
        R = np.matrix(port.iloc[id, :7]).T
        rf = df_FF.iloc[id, -1]
        
        w0= [lb,lb,lb,lb,lb,lb,lb]
        
        consA = ({'type': 'eq', 'fun': cons1})
        consB = ({'type': 'ineq', 'fun': lambda x: x})
        cons = [consA, consB]
        res= minimize(calculate_portfolio_var, w0, args=V, method='SLSQP',constraints=cons)
        # Dyamnic Weights
        w_g = res.x
        # Dynamic Portfolio Returns
        port_g = w_g*R
        
        Vola = np.sqrt(np.dot(w_g*V,w_g))
        
        # Static weights
        EqriskW = np.array([lb,lb,lb,lb,lb,lb,lb])
        
        Wportfolio[num].iloc[id, :] = w_g
        port.iloc[id,  7: ] = port_g, Vola, EqriskW*R 
    
    plt.figure(figsize = (13, 10))
    plt.plot(port.dropna().index, port.dropna().iloc[:,-3].cumsum(),'r-.',label=port_title[num]+' Dynamic Weighting')
    plt.plot(port.dropna().index, port.dropna().iloc[:,-1].cumsum(),'b-.',label=port_title[num]+' Equal-Risk Weighting')
    plt.title(port_title[num]+" Dynamic"+" vs "+port_title[num]+" Static equally weighted")
#    x = port.dropna().index
#    plt.xlim(x[0], x[-1])
    plt.legend() 
    plt.show()
    plt.savefig('Part2A'+port_title[num]+'EqualRiskContribution.png')

# Weights are df_DWVa df_DWEq
# Returns are df_Va, df_Eq
#%% Ratios for Equal Risk Contributions Portfolios

Pt2Ratiosind = ['Dynamic-Weight Value Portfolio',
                'Dynamic-Weight Equal Portfolio',
                'Static-Weight Value Portfolio',
                'Static-Weight Equal Portfolio']

Pt2Ratioscol = ['Volatility', 'Mean-Return', 'Sharpe-Ratio','MaxDrawdown']

Pt2Ratios = pd.DataFrame(index = Pt2Ratiosind , columns = Pt2Ratioscol, dtype='float')

for num, port in enumerate(portfolio):
    shape = len(port.iloc[: ,-3].dropna())
    # Volatility
    Pt2Ratios.iloc[num , 0] =  port.iloc[: ,-3].dropna().std()
    Pt2Ratios.iloc[num+2 , 0] =  port.iloc[: ,-1].dropna().std()
    # Mean Return
    Pt2Ratios.iloc[num , 1] =  np.mean(port.iloc[: ,-3].dropna())
    Pt2Ratios.iloc[num+2 , 1] =  np.mean(port.iloc[: ,-1].dropna())
    # Sharpe Ratio
    Pt2Ratios.iloc[num , 2] =  np.mean(port.iloc[: ,-3].dropna()-df_FF['RF'][-shape:])/port.iloc[: ,-3].std()
    Pt2Ratios.iloc[num+2 , 2] =  np.mean(port.iloc[: ,-1].dropna()-df_FF['RF'][-shape:])/port.iloc[: ,-1].std()
    # Max Drawdown
    Pt2Ratios.iloc[num , 3] =  MaxDrawndown(port.iloc[: ,-3].dropna().cumsum())
    Pt2Ratios.iloc[num+2 , 3] =  MaxDrawndown(port.iloc[: ,-1].dropna().cumsum())

print("Equal Risk Contributions Portfolios ")
print(Pt2Ratios)

#%% Part III
#%% FACTOR Persistance Backtest

df_PVa = df_Va.iloc[: , :7]
df_PVa = df_PVa.assign(EW=np.nan)
df_PVa['EW'] = (df_Va.iloc[: , :7]*1/7).sum(axis=1)

df_PEq = df_Eq.iloc[: , :7]
df_PEq = df_PVa.assign(EW=np.nan)
df_PEq['EW'] = (df_Eq.iloc[: , :7]*1/7).sum(axis=1)

Ptype = [df_PVa, df_PEq ]
type = ['Value', 'Equal']
# Plotting
#plt.title('Factor Persistence')
#for num, style in enumerate(type):
num=0
plt.figure(figsize = (15, 10))
for col in df_PVa:
    x = Ptype[num].loc[:, col].cumsum()[-1]
    benchmark = Ptype[num].loc[: ,'EW'].cumsum()[-1]
    if col == 'EW':
        plt.plot(Ptype[num].loc[:, col].cumsum() ,'r-', label = col + " " + type[num]) 
        plt.legend()
    elif x> benchmark:
        plt.plot(Ptype[num].loc[:, col].cumsum() ,'-', label = col + " " + type[num])
        plt.legend()
plt.title('Factor Persistence')
plt.xlim(df_PVa.dropna().index[0],df_PVa.dropna().index[-1])
plt.show()
plt.savefig('Part3A '+ type[num] +'FactorPersistance.png')

#%% PCA
PCA_COV =df_PVa.cov()
print(PCA_COV)

X = np.asarray(df_PVa)
[n,m] = X.shape

pca = PCA(n_components=3) # number of principal components
pca.fit(X)

percentage =  pca.explained_variance_ratio_
percentage_cum = np.cumsum(percentage)
print('{0:.2f}% of the variance is explained by the first 2 PCs'.format(percentage_cum[-1]*100)) 

pca_components = pca.components_

print(pca_components)

xx = np.arange(1,len(percentage)+1,1)

sns.set_context('talk')
plt.figure(figsize = (7, 12))
plt.bar(xx, percentage*100, align = "center")
plt.title('Contribution of principal components',fontsize = 16)
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.xticks(xx,fontsize = 16) 
plt.yticks(fontsize = 16)
plt.xlim([0, 3+1])
plt.ylim(0, 50)
#%% 
# Optimal Sharpe Portfolio
Opt_port12 = pd.DataFrame({'VBP':df_Va['BP'], 'VPrior_12-1':df_Va['Prior_12-1'],
                         'VPrior_1':df_Va['Prior_1'],'EBP':df_Eq['BP'], 
                         'EPrior_12-1':df_Eq['Prior_12-1'],'EPrior_1':df_Eq['Prior_1']})
"""
# VIX index    
df_VIX = pd.read_csv('VIX.csv')
df_VIX['Date'] = pd.to_datetime(df_VIX['Date'])
df_VIX.set_index('Date',inplace = True)
"""    
# Rolling 12 months std
Opt_port12['MKT-RF'] = df_FF['MKT-RF']
Opt_port12['MKT-RF-Vol'] = Opt_port12['MKT-RF'].shift(1).rolling(12).std().dropna()

# Rolling 3 months std
Opt_port3 = Opt_port12.iloc[:, :-1]
Opt_port3['MKT-RF-Vol'] = Opt_port3['MKT-RF'].shift(1).rolling(3).std().dropna()

# Rolling 6 months std
Opt_port6 = Opt_port12.iloc[:, :-1]
Opt_port6['MKT-RF-Vol'] = Opt_port6['MKT-RF'].shift(1).rolling(6).std().dropna()


# Rolling 24 months std
Opt_port24 = Opt_port12.iloc[:, :-1]
Opt_port24['MKT-RF-Vol'] = Opt_port24['MKT-RF'].shift(1).rolling(24).std().dropna()

# Rolling 36 months std
Opt_port36 = Opt_port12.iloc[:, :-1]
Opt_port36['MKT-RF-Vol'] = Opt_port36['MKT-RF'].shift(1).rolling(36).std().dropna()

# Plot VIX timeseries# Plot  
def RollingVol(Optimal, Lookback, type):
    plt.figure(figsize = (17, 5))
    ax = Optimal.loc['1990-01-01 00:00:00':  ,'MKT-RF-Vol'].plot(color='black')
    
    # The threshold for dividing high/low vol regimes is the long-term mean of S&P 500 volatility.
    if type==12:
        Mktthreshold =  Optimal.loc[:'1990-01-01 00:00:00' ,'MKT-RF-Vol'].mean() \
                        +1*Optimal.loc[:'1990-01-01 00:00:00', 'MKT-RF-Vol'].std()
    elif type == 6:
        Mktthreshold =  Optimal.loc[:'1990-01-01 00:00:00' ,'MKT-RF-Vol'].mean() \
                        +1*Optimal.loc[:'1990-01-01 00:00:00', 'MKT-RF-Vol'].std()
    elif type == 3:
        Mktthreshold =  Optimal.loc[:'1990-01-01 00:00:00' ,'MKT-RF-Vol'].mean() \
                        +2*Optimal.loc[:'1990-01-01 00:00:00', 'MKT-RF-Vol'].std()                
    elif type == 24 :
        Mktthreshold =  Optimal.loc[:'1990-01-01 00:00:00' ,'MKT-RF-Vol'].mean() \
                        +0.5*Optimal.loc[:'1990-01-01 00:00:00', 'MKT-RF-Vol'].std()
    else:
        Mktthreshold =  Optimal.loc[:'1990-01-01 00:00:00' ,'MKT-RF-Vol'].mean() \
                        +0.5*Optimal.loc[:'1990-01-01 00:00:00', 'MKT-RF-Vol'].std()                
    # Index[318] = 1990-01-01 date
    # Highlight regions of high volatility.
    Mktthreshold = np.float32(Mktthreshold)
    x = Optimal.loc['1990-01-01 00:00:00':,'MKT-RF-Vol'].index
    ymax = Opt_port12.loc['1990-01-01 00:00:00':,'MKT-RF-Vol'].max() + 5
    ax.fill_between(x, 0, ymax, where=Optimal.loc['1990-01-01 00:00:00':,'MKT-RF-Vol'] > Mktthreshold,\
                            facecolor='green', alpha=0.5, interpolate=True)
    
    # Add additional styling.
    plt.xlim(x[0], x[-1])
    plt.ylim(0, ymax)
    ax.set(title='Market Volatility '+Lookback, ylabel='Volatility')
    plt.savefig('Part3B '+Lookback+'.png')

#    return Optimal.loc['1990-01-01 00:00:00':,'MKT-RF-Vol'] > Mktthreshold
    return Mktthreshold

#xx = Mktthreshold6

Mktthreshold3 = RollingVol(Opt_port3, 'Lookback 3 months', 3)
Mktthreshold6 = RollingVol(Opt_port6, 'Lookback 6 months', 6)
Mktthreshold12 = RollingVol(Opt_port12, 'Lookback 12 months', 12)
Mktthreshold24 = RollingVol(Opt_port24, 'Lookback 24 months', 24)
Mktthreshold36 = RollingVol(Opt_port36, 'Lookback 36 months', 36)

Opt_port3 = Opt_port3.assign(Returns=np.nan, Regime=np.nan, EW=np.nan)
WOpt_port3 = pd.DataFrame(index=Opt_port3.index,columns =Opt_port3.columns[:6],dtype = 'float')

Opt_port6 = Opt_port12.assign(Returns=np.nan, Regime=np.nan, EW=np.nan)
WOpt_port6 = pd.DataFrame(index=Opt_port6.index,columns =Opt_port6.columns[:6],dtype = 'float')

Opt_port12 = Opt_port12.assign(Returns=np.nan, Regime=np.nan, EW=np.nan)
WOpt_port12 = pd.DataFrame(index=Opt_port12.index,columns =Opt_port12.columns[:6],dtype = 'float')

Opt_port24 = Opt_port24.assign(Returns=np.nan, Regime=np.nan, EW=np.nan)
WOpt_port24 = pd.DataFrame(index=Opt_port24.index,columns =Opt_port24.columns[:6],dtype = 'float')

Opt_port36 = Opt_port36.assign(Returns=np.nan, Regime=np.nan, EW=np.nan)
WOpt_port36 = pd.DataFrame(index=Opt_port36.index,columns =Opt_port36.columns[:6],dtype = 'float')

#%%  Backtest strategy

# min var optimization
def MinVarMaxSharpe(w,V):
    w = np.matrix(w)
    if BTPort['MKT-RF-Vol'][id] > threshold:
        return np.sqrt(w*Cov_matrix*w.T)
    else:
        ExRet = np.matrix([BTPort.iloc[id-12:id, col]-df_FF.iloc[id-12:id, -1] for col in range(6)])
        SR_upper = np.mean(w*ExRet)
        return -SR_upper/np.sqrt(w*Cov_matrix*w.T)
    
# Dollar neutral
def DollarNeutral(w):
    return np.sum(w)-1

bndsA = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))

#Backtest = [Opt_port3, Opt_port6,Opt_port12,Opt_port24,Opt_port36]
#BacktestW = [WOpt_port3, WOpt_port6,WOpt_port12,WOpt_port24,WOpt_port36]
#BacktestTH = [Mktthreshold3, Mktthreshold6,Mktthreshold12,Mktthreshold24,Mktthreshold36]

Backtest = [Opt_port6,Opt_port12]
BacktestW = [WOpt_port6,WOpt_port12]
BacktestTH = [Mktthreshold6,Mktthreshold12]


BT = [3,6,12,24,36]
for num,BTPort in enumerate(Backtest):
    roll = BT[num]
    threshold = BacktestTH[num]
    print(threshold)
    for id in range(318, len(BTPort)):# Index[318] = 1990-01-01 date
        # Covariance matrix
        cov  = BTPort.iloc[id-roll:id , :6].cov() 
        Cov_matrix = np.matrix(cov)
        # Portfolio Returns
        R = np.matrix(BTPort.iloc[id, :6]).T
        # Risk-Free Rate
        rf = df_FF.iloc[id, -1]
            
        # Initial Guess for Weights
        w0= [lb,lb,lb,lb,lb,lb]
        
        # Unconstrained portfolio
        consX = ({'type': 'eq', 'fun': cons1})
        consY = ({'type': 'ineq', 'fun': lambda x: x}) # Only Long leg exposure
        cons = [consX, consY]
        res= minimize(MinVarMaxSharpe, w0, args=V,bounds=bndsA, method='SLSQP',constraints=cons)
        w_g = res.x
        # Portfolio Returns
        BTPort.iloc[id, 8] = w_g*R
        # Regime Change
        if BTPort['MKT-RF-Vol'][id] >=threshold:
            BTPort.iloc[id, 9] = 0
        else: 
            BTPort.iloc[id, 9] = 1
            
        BacktestW[num].iloc[id,:] = w_g
        EqualWeight = np.matrix(w0)
        BTPort.iloc[id, 10] = EqualWeight*R
    
#    BTPort.iloc[:, 8] = BTPort.iloc[id, 8].shift(1)
#    BTPort.iloc[318, 8] = 0
    
#    BacktestW[num]['Regime'] = BTPort['Regime']

#print('Portfolio Returns:',Opt_port36['Returns'].dropna().cumsum()[-1] )
#print('Portfolio Sharpe:',np.mean((Opt_port36['Returns']-df_FF['RF']).dropna())/Opt_port36['Returns'].dropna().std() )
#print('Portfolio Volatility:',Opt_port36['Returns'].dropna().std() )
#
#print('EW Portfolio Sharpe:',np.mean((Opt_port36['EW']-df_FF['RF']).dropna())/Opt_port36['EW'].dropna().std() )
#print('Portfolio Volatility:',Opt_port36['EW'].dropna().std() )

    
    # Plot  
    plt.figure(figsize = (15, 5))
    ax = plt.plot(BTPort['Returns'].dropna().index,BTPort['Returns'].dropna().cumsum(),color='black',label= str(roll) +' Lookback '+'Regime Portfolio')
    plt.plot(BTPort['Returns'].dropna().index,BTPort['EW'].dropna().cumsum(),color='blue',label='EW')
    
    # Highlight regions of high volatility.
    x = BTPort['Returns'].dropna().index
    ymax = BTPort['Returns'].dropna().cumsum().max() + 5
    plt.fill_between(x, 0, ymax, where=BTPort.loc['1990-01-01 00:00:00': ,'MKT-RF-Vol']>threshold, facecolor='green', alpha=0.5, interpolate=True)
    
    # Add additional styling.
    plt.xlim(x[0], x[-1])
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.title(str(roll) + ' Lookback '+'Portfolio Returns')
    plt.savefig('Part3C '+str(roll) + ' Lookback '+'Portfolio Returns'+'.png')

#plt.figure(figsize = (15, 5))
#for num,BTPort in enumerate(Backtest):
#    roll = BT[num]
#    threshold = BacktestTH[num]
#    ax = plt.plot(BTPort['Returns'].dropna().index,BTPort['Returns'].dropna().cumsum(),label= str(roll) +' Lookback '+'Regime Portfolio')
#plt.plot(Opt_port6['Returns'].dropna().index,Opt_port6['EW'].dropna().cumsum(),color='blue',label='EW')
#x = Opt_port6['Returns'].dropna().index
#plt.xlim(x[0], x[-1])
#plt.xlabel('Date')
#plt.ylabel('Cumulative Returns')
#plt.legend()
#plt.title('Comparsion of Portfolio Returns')
#plt.savefig('Part3D Comparsion Portfolio Returns'+'.png')

#%% Part3Ratios

Pt3Ratiosind = ['3 months Lookback Portfolio',
                '6 months Lookback Portfolio',
                '12 months Lookback Portfolio',
                '24 months Lookback Portfolio',
                '36 months Lookback Portfolio',
                'Equal Weighted Portfolio']

Pt3Ratioscol = ['Volatility', 'Mean-Return', 'Sharpe-Ratio','MaxDrawdown']

Pt3Ratios = pd.DataFrame(index = Pt3Ratiosind , columns = Pt3Ratioscol, dtype='float')
BacktestA = [Opt_port3, Opt_port6,Opt_port12,Opt_port24,Opt_port36]
for num, port in enumerate(BacktestA):
    shape = len(port.iloc[: ,-3].dropna())
    # Volatility
    Pt3Ratios.iloc[num , 0] =  port.iloc[318: ,8].dropna().std()
    # Mean Return
    Pt3Ratios.iloc[num , 1] =  np.mean(port.iloc[318: ,8].dropna())
    # Sharpe Ratio
    Pt3Ratios.iloc[num , 2] =  np.mean(port.iloc[318: ,8].dropna()-df_FF['RF'][-318:])/port.iloc[318: ,8].std()
    # Max Drawdown
    Pt3Ratios.iloc[num , 3] =  MaxDrawndown(port.iloc[318: ,8].dropna().cumsum())

Pt3Ratios.iloc[num+1 , 0] =  Opt_port6.iloc[318: ,-1].dropna().std()
# Mean Return
Pt3Ratios.iloc[num+1 , 1] =  np.mean(Opt_port6.iloc[318: ,-1].dropna())
# Sharpe Ratio
Pt3Ratios.iloc[num+1 , 2] =  np.mean(Opt_port6.iloc[318: ,-1].dropna()-df_FF['RF'][-318:])/Opt_port6.iloc[318: ,-3].std()
# Max Drawdown
Pt3Ratios.iloc[num+1 , 3] =  MaxDrawndown(Opt_port6.iloc[318: ,-1].dropna().cumsum())

print("Adaptive Portfolios ")
print(Pt3Ratios)

#%%

plt.figure(figsize = (15, 5))
ax = plt.plot(Opt_port6['Returns'].dropna().index,Opt_port6['Returns'].dropna().cumsum(),color='black',label= str(6) +' Lookback '+'Regime Portfolio')
plt.plot(Opt_port6['Returns'].dropna().index,Opt_port6['EW'].dropna().cumsum(),color='blue',label='EW')

# Highlight regions of high volatility.
x = Opt_port6['Returns'].dropna().index
ymax = Opt_port6['Returns'].dropna().cumsum().max() + 5
plt.fill_between(x, 0, ymax+20, where=xx == True, facecolor='green', alpha=0.5, interpolate=True)

# Add additional styling.
plt.xlim(x[0], x[-1])
plt.ylim(-5, ymax+20)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title(str(6) + ' Lookback '+'Portfolio Returns')
plt.savefig('Part3C '+str(6) + ' Lookback '+'Portfolio Returns'+'.png')

#x = Opt_port6.loc['1990-01-01 00:00:00': ,'MKT-RF-Vol']>threshold