# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:53:09 2017

@author: Jon Wee
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as sm

#### Excel files
index = 'KOSPI--Index.csv'
bank = '024110-KS-Equity.csv'
non_bank = '017670-KS-Equity.csv'

#############PART ONE###############################################################
def data_sampling(X):
    data = pd.read_csv(X)
    date = data['Date']
    data_date = [pd.to_datetime(date, format='%Y-%m-%d') for date in date]
    data.index = data_date
    data = data.drop('Date', axis=1)    
    Num_row = len(data)
    i = 0
    while i < Num_row :
        p = data['PX_LAST'][i]
        if np.isnan(p):
            data.drop(data.index[i], inplace=True) 
            print("Deleted %s" % (data.index[i]))
            Num_row  -= 1
        i += 1
#    print("Number of rows AFTER filtering = %d" % (n),"\n")
####################################################################################
    
    daily = data['PX_LAST']
    monthly = daily.resample('M').last()
    quarterly = daily.resample('Q').last()
    annually = daily.resample('A').last()
    
    ret_daily = (daily/daily.shift(1))-1
    ret_daily = ret_daily[1:]
    ret_monthly = (monthly/monthly.shift(1))-1
    ret_monthly = ret_monthly[1:]
    ret_quarterly = (quarterly/quarterly.shift(1))-1
    ret_quarterly = ret_quarterly[1:]
    ret_annually  = (annually/annually.shift(1))-1
    ret_annually = ret_annually[1:]
    
    m_ret_daily = ret_daily.mean()
    v_ret_daily = ret_daily.var()

    
    k = [1.5,2,3,4]
    PC = []
    LWR_B = []
    UPP_B = []
    PA = []
    
    for i in k:
        PC = np.append(PC, (1-1/(i**2)))
        LWR_B = np.append(LWR_B, m_ret_daily-i*(v_ret_daily**0.5))
        UPP_B = np.append(UPP_B, m_ret_daily+i*(v_ret_daily**0.5))
        count = 0
        for j in range(len(ret_daily)):
            if LWR_B[-1]<ret_daily[j]:
                if UPP_B[-1]>ret_daily[j]:
                    count= count+1
        PA = np.append(PA, count/len(ret_daily))
    
     
        
    return PC,LWR_B, UPP_B, PA, ret_daily, ret_monthly, ret_quarterly, ret_annually, monthly    

##### BANK  ###################################################################
prob_c_bank,lower_b_bank, upper_b_bank, prob_a_bank,\
ret_daily_bank, ret_monthly_bank, ret_quarterly_bank, ret_annually_bank, monthly_bank = data_sampling(bank)     

##### NON BANK  ###################################################################
prob_c_nonbank,lower_b_nonbank, upper_b_nonbank, prob_a_nonbank,\
ret_daily_nonbank, ret_monthly_nonbank, ret_quarterly_nonbank, ret_annually_nonbank, monthly_nonbank = data_sampling(non_bank)     

##### Index ###################################################################    
prob_c_index,lower_b_index, upper_b_index, prob_a_index,\
ret_daily_index, ret_monthly_index, ret_quarterly_index, ret_annually_index, monthly_index = data_sampling(index)     


###### cheby             #############################################################
Cheb_bank = pd.DataFrame([prob_c_bank,lower_b_bank, upper_b_bank, prob_a_bank], index=['PC','Lower bound', 'Upper bound', 'PA'] ,columns = ['1.5','2' ,'3' ,'4'])
Cheb_bank = Cheb_bank.transpose()

Cheb_non_bank = pd.DataFrame([prob_c_nonbank,lower_b_nonbank, upper_b_nonbank, prob_a_nonbank], index=['PC','Lower bound', 'Upper bound', 'PA'] ,columns = ['1.5','2' ,'3' ,'4'])
Cheb_non_bank = Cheb_non_bank.transpose()

Cheb_index = pd.DataFrame([prob_c_index,lower_b_index, upper_b_index, prob_a_index], index=['PC','Lower bound', 'Upper bound', 'PA'] ,columns = ['1.5','2' ,'3' ,'4'])
Cheb_index = Cheb_index.transpose()

print('Chebyshev of Industrial Bank of Korea')
print(Cheb_bank)
print(' ')
print('Chebyshev of SK telecom')
print(Cheb_non_bank)
print(' ')
print('Chebyshev of Korea index')
print(Cheb_index)
print(' ')
###### Bank Covariance   #############################################################
D_Cov_Index_Bank = pd.DataFrame([ret_daily_bank,ret_daily_index])
D_Cov_Index_Bank = D_Cov_Index_Bank.transpose()
D_Cov_Index_Bank = D_Cov_Index_Bank.cov()   
D_Cov_Index_Bank.index = [0,1] 
D_Cov_Index_Bank.columns = ['bank','index']
Covariance_D_Index_Bank = D_Cov_Index_Bank['bank'][1]

M_Cov_Index_Bank = pd.DataFrame([ ret_monthly_bank,ret_monthly_index])
M_Cov_Index_Bank = M_Cov_Index_Bank.transpose()
M_Cov_Index_Bank = M_Cov_Index_Bank.cov()  
M_Cov_Index_Bank.index = [0,1] 
M_Cov_Index_Bank.columns = ['bank','index']  
Covariance_M_Index_Bank = M_Cov_Index_Bank['bank'][1]

Q_Cov_Index_Bank = pd.DataFrame([ret_quarterly_bank,ret_quarterly_index])
Q_Cov_Index_Bank = Q_Cov_Index_Bank.transpose()
Q_Cov_Index_Bank = Q_Cov_Index_Bank.cov() 
Q_Cov_Index_Bank.index = [0,1] 
Q_Cov_Index_Bank.columns = ['bank','index']    
Covariance_Q_Index_Bank = Q_Cov_Index_Bank['bank'][1]

A_Cov_Index_Bank = pd.DataFrame([ret_annually_bank,ret_annually_index])
A_Cov_Index_Bank = A_Cov_Index_Bank.transpose()
A_Cov_Index_Bank = A_Cov_Index_Bank.cov() 
A_Cov_Index_Bank.index = [0,1] 
A_Cov_Index_Bank.columns = ['bank','index']     
Covariance_A_Index_Bank = A_Cov_Index_Bank['bank'][1]

Total_Cov_Index_bank = pd.DataFrame([Covariance_D_Index_Bank, Covariance_M_Index_Bank, Covariance_Q_Index_Bank, Covariance_A_Index_Bank ]\
                                    ,index = ['Daily','Monthly', 'Quarterly', 'Annually'])
print('Covariance of Industrial Bank of Korea')
print(Total_Cov_Index_bank)
print(' ')
#####  nonBANK covariance ###########################################################
D_Cov_Index_nonBank = pd.DataFrame([ret_daily_nonbank,ret_daily_index])
D_Cov_Index_nonBank = D_Cov_Index_nonBank.transpose()
D_Cov_Index_nonBank = D_Cov_Index_nonBank.cov()
D_Cov_Index_nonBank.index = [0,1] 
D_Cov_Index_nonBank.columns = ['nonbank','index'] 
Covariance_D_Index_nonBank = D_Cov_Index_nonBank['nonbank'][1]

M_Cov_Index_nonBank = pd.DataFrame([ret_monthly_nonbank,ret_monthly_index])
M_Cov_Index_nonBank = M_Cov_Index_nonBank.transpose()
M_Cov_Index_nonBank = M_Cov_Index_nonBank.cov()
M_Cov_Index_nonBank.index = [0,1] 
M_Cov_Index_nonBank.columns = ['nonbank','index'] 
Covariance_M_Index_nonBank = M_Cov_Index_nonBank['nonbank'][1]

Q_Cov_Index_nonBank = pd.DataFrame([ret_quarterly_nonbank,ret_quarterly_index])
Q_Cov_Index_nonBank = Q_Cov_Index_nonBank.transpose()
Q_Cov_Index_nonBank = Q_Cov_Index_nonBank.cov()
Q_Cov_Index_nonBank.index = [0,1] 
Q_Cov_Index_nonBank.columns = ['nonbank','index'] 
Covariance_Q_Index_nonBank = Q_Cov_Index_nonBank['nonbank'][1]

A_Cov_Index_nonBank = pd.DataFrame([ret_annually_nonbank,ret_annually_index])
A_Cov_Index_nonBank = A_Cov_Index_nonBank.transpose()
A_Cov_Index_nonBank = A_Cov_Index_nonBank.cov()
A_Cov_Index_nonBank.index = [0,1] 
A_Cov_Index_nonBank.columns = ['nonbank','index'] 
Covariance_A_Index_nonBank = A_Cov_Index_nonBank['nonbank'][1]    

Total_Cov_Index_nonbank = pd.DataFrame([Covariance_D_Index_nonBank,Covariance_M_Index_nonBank,Covariance_Q_Index_nonBank,Covariance_A_Index_nonBank],\
                                       index=['Daily','Monthly', 'Quarterly', 'Annually'])
  
Total_Cov = pd.concat([Total_Cov_Index_bank,Total_Cov_Index_nonbank], axis=1)  
Total_Cov = Total_Cov.transpose()

print('Covariance of SK telecom')
print(Total_Cov_Index_nonbank)
print(' ')
#####  BANK RIGHT LEFT COSKEWNESS ###########################################################    
def co_skewness(bank, index, type):
    if type == 'L':
        skew = ((bank-bank.mean())**2 * (index-index.mean())).mean()/  \
                (bank.var()*index.std())
        return skew       
    elif type =='R':
        skew = ((bank-bank.mean()) * (index-index.mean())**2).mean()/  \
                (bank.var()*index.std())
        return skew

bank = [ret_daily_bank, ret_monthly_bank, ret_quarterly_bank, ret_annually_bank]
index = [ret_daily_index, ret_monthly_index, ret_quarterly_index, ret_annually_index]
nonbank = [ret_daily_nonbank, ret_monthly_nonbank, ret_quarterly_nonbank, ret_annually_nonbank]
L_coskew_index_bank = []
R_coskew_index_bank = []
L_coskew_index_nonbank = []
R_coskew_index_nonbank = []
for i in range(4): 
    L_coskew_index_bank = np.append(L_coskew_index_bank, co_skewness(bank[i],index[i],'L') )  
    R_coskew_index_bank = np.append(R_coskew_index_bank,co_skewness(bank[i],index[i],'R' ) )
    L_coskew_index_nonbank = np.append(L_coskew_index_nonbank, co_skewness(nonbank[i],index[i],'L') )
    R_coskew_index_nonbank = np.append(R_coskew_index_nonbank, co_skewness(nonbank[i],index[i],'R') )

Coskew_index_bank = pd.DataFrame([L_coskew_index_bank, R_coskew_index_bank], index=['Left coskew','Right coskew'], columns = ['Daily','Monthly','Quarterly','Annually'])
Coskew_index_nonbank = pd.DataFrame([L_coskew_index_nonbank, R_coskew_index_nonbank], index=['Left coskew','Right coskew'], columns = ['Daily','Monthly','Quarterly','Annually'])

print('Coskewness of Industrial Bank of Korea')
print(Coskew_index_bank)
print(' ')
print('Coskewness of SK telecom')
print(Coskew_index_nonbank)
print(' ')
#####    Regression            ##############################################################

#ret_monthly_bank, ret_monthly_index, ret_monthly_nonbank
  
def Data_Rf(X):
    data = pd.read_csv(X)
    date = data['Date']
    data_date = [pd.to_datetime(date, format='%Y-%m-%d') for date in date]
    data.index = data_date
    data = data.drop('Date', axis=1)    
    Num_row = len(data)
    i = 0
    while i < Num_row :
        p = data['PX_LAST'][i]
        if np.isnan(p):
            data.drop(data.index[i], inplace=True) 
            print("Deleted %s" % (data.index[i]))
            Num_row  -= 1
        i += 1
#    print("Number of rows AFTER filtering = %d" % (n),"\n")
####################################################################################
    
    daily = data['PX_LAST']
    monthly = 0.01*daily.resample('M').last()/12.0  
    
    return monthly

Rf = Data_Rf('KRBO1M--Index.csv')

Data_monthly = pd.DataFrame([ ret_monthly_bank,ret_monthly_nonbank,ret_monthly_index, Rf], index=['bank','nonbank','index','Rf'] )
Data_monthly = Data_monthly.transpose()
#Data_monthly = Data_monthly.dropna(axis=0, how='any')


Data_bank = Data_monthly['bank']-Data_monthly['Rf'].shift(1)
Data_bank =Data_bank.dropna(axis=0, how='any')
Data_index = Data_monthly['index']-Data_monthly['Rf'].shift(1)
Data_index=Data_index.dropna(axis=0, how='any')
Data_nonbank = Data_monthly['nonbank']-Data_monthly['Rf'].shift(1)
Data_nonbank=Data_nonbank.dropna(axis=0, how='any')
"""
RET_Rf = pd.concat([Data_bank,Data_index,Data_nonbank], axis=1)


model1 = sm.ols('RET_Rf.iloc[:,0] ~ RET_Rf.iloc[:,1]',data=RET_Rf).fit()
print(model1.summary())

model2 = sm.ols('RET_Rf.iloc[:,2] ~ RET_Rf.iloc[:,1]',data=RET_Rf).fit()
print(model2.summary())
#x=Data_bank
#y=Data_index
#model = sm.ols('y~x', data=dat4).fit()


"""

def CAPM(y, x, title,xlabel,ylabel):
    
    xa = np.mean(x)
    ya = np.mean(y)
    
    xd = x - xa
    yd = y - ya
    
    n = len(xd)
    
    # estimates
    beta = sum(xd*yd)/sum(xd*xd)
    alpha = ya - beta*xa
    
    # fitted value yf and residual uh
    yf = alpha + beta * x
    uh = y - yf
    
    # Total sum of squares and residual sum of squares
    tss = sum(yd**2)
    rss = sum(uh**2)
    
    # R square
    R2 = 1 - rss/tss
    
    # variance of residuals
    sigma_uh2 = rss/(n-2)
    
    # standard errors
    xss = sum(xd**2)
    SE_beta = np.sqrt(sigma_uh2/xss)
    SE_alpha = np.sqrt(sigma_uh2 * (1/n + (xa**2)/xss))
    
    # t statistics
    tstatalpha = alpha/SE_alpha
    tstatbeta = beta/SE_beta
    
    # critical value for t statistics
    t_critical = stats.t.ppf(1-0.025, n-2)
    
    # two-sided pvalue 
    pvaluebeta = stats.t.sf(np.abs(tstatbeta), n-2)*2 
    pvaluealpha = stats.t.sf(np.abs(tstatalpha), n-2)*2 
    
    # F statistic
    msr = sum((yf-ya)**2)
    F = msr/sigma_uh2
    # critical value for F statistic
    F_critical = stats.f.ppf(1-0.025, 1, n-2)
    
    plt.figure(figsize=(8 , 6))
    # plot the fitted line and data
#    xl = np.linspace(0, 700, 200)
#    yl = b0h + b1h*xl+
    plt.plot(x, yf)
    plt.plot(x, y, 'o')
    plt.title('Monthly Returns Regression: %s' %title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    return beta, alpha, SE_beta, SE_alpha, R2, tstatalpha,\
 tstatbeta, t_critical, pvaluebeta, pvaluealpha, F, F_critical
#Data_nonbank, Data_bank, Data_index    
#bank_beta, bank_alpha, bank_SE_beta, bank_SE_alpha, bank_tstatalpha, bank_tstatbeta, bank_t_critical, bank_pvaluebeta, bank_pvaluealpha, bank_F, bank_F_critical\

bank = CAPM(Data_bank, Data_index, 'Industrial bank of korea vs KOSPI', 'KOSPI Excess returns','Industrial bank of korea Excess returns')
bank = pd.DataFrame([bank], index=['Industrial Bank of Korea'])
bank = bank.transpose()
bank.index = ['beta', 'alpha', 'SE_beta', 'SE_alpha', 'R2', 'tstatalpha', 'tstatbeta', 't_critical', 'pvaluebeta', 'pvaluealpha', 'F', 'F_critical']
print('Industrial Bank of Korea Regression')
print(bank)
#nonbank_beta, nonbank_alpha, nonbank_SE_beta, nonbank_SE_alpha, nonbank_tstatalpha, nonbank_tstatbeta, nonbank_t_critical, nonbank_pvaluebeta, nonbank_pvaluealpha, nonbank_F, nonbank_F_critical\
nonbank = CAPM(Data_nonbank, Data_index, 'SK Telecom vs KOSPI', 'KOSPI Excess returns','SK Telecom Excess Returns')
nonbank = pd.DataFrame([nonbank], index=['SK telecom'])
nonbank = nonbank.transpose()
nonbank.index = ['beta', 'alpha', 'SE_beta', 'SE_alpha', 'R2', 'tstatalpha', 'tstatbeta', 't_critical', 'pvaluebeta', 'pvaluealpha', 'F', 'F_critical']
print('SK telecom Regression')
print(nonbank)


"""
Y = Data_index
X = Data_nonbank
X = sm.add_constant(X)

model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
"""
