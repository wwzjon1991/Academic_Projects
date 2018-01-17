# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 04:44:19 2017

@author: Jon Wee
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#### Excel files
index = 'KOSPI--Index.csv'
bank = '024110-KS-Equity.csv'
non_bank = '017670-KS-Equity.csv'
############# Functions    #########################################################
####################################################################################
############# T-test       #########################################################
def t_test(r,hypo):
    return (r.mean()-hypo)/(np.sqrt(np.var(r,ddof=1)/len(r)))
############# Chi-Square-test       ################################################
def chi_square(r,v):
    return (len(r)-1)*(np.var(r,ddof=1)/v)
############# JB Test        #######################################################
def skewness(z):
   n = len(z)
   g1 = sum(z**3)
   G1 = g1*n/((n-1)*(n-2))
   return G1

def kurtosis(z):
   n = len(z)
   g2 = sum(z**4)
   
   G2 = g2 *n*(n+1)/( (n-1)*(n-2)*(n-3) )
   return G2

def jb_test(skewness, kurt, n):
   return (n/6.0) * (skewness**2 + ((kurt-3)**2)/4.0)  



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
#    return daily, monthly, quarterly, annually
#daily, monthly, quarterly, annually = data_sampling(non_bank)    
####     log returns        ########################################################
    log_daily = np.log(daily) - np.log(daily.shift(1))
    log_daily = log_daily[1:]
    log_monthly = np.log(monthly) - np.log(monthly.shift(1))
    log_monthly = log_monthly[1:]
    log_quarterly = np.log(quarterly) - np.log(quarterly.shift(1))
    log_quarterly  = log_quarterly[1:]
    log_annually = np.log(annually) - np.log(annually.shift(1))
    log_annually = log_annually[1:]
    
    return log_daily, log_monthly, log_quarterly, log_annually

#############    NON BANK STATISTIC        ########################################################  
log_daily, log_monthly, log_quarterly, log_annually = data_sampling(non_bank)
T_test_nonbank = []
T_test_nonbank = np.append(T_test_nonbank, t_test(log_daily,0))
T_test_nonbank = np.append(T_test_nonbank, t_test(log_monthly, 22*log_daily.mean()))
T_test_nonbank = np.append(T_test_nonbank, t_test(log_quarterly, 3*log_monthly.mean()))
T_test_nonbank = np.append(T_test_nonbank, t_test(log_annually, 4*log_quarterly.mean()))

Ttest_pvalue_nonbank = []
Ttest_pvalue_nonbank  = np.append(Ttest_pvalue_nonbank, (1 - st.t.cdf(abs(T_test_nonbank[0]), len(log_daily)-2))*2)
Ttest_pvalue_nonbank  = np.append(Ttest_pvalue_nonbank, (1 - st.t.cdf(abs(T_test_nonbank[1]), len(log_monthly)-2))*2)
Ttest_pvalue_nonbank  = np.append(Ttest_pvalue_nonbank, (1 - st.t.cdf(abs(T_test_nonbank[2]), len(log_quarterly)-2))*2)
Ttest_pvalue_nonbank  = np.append(Ttest_pvalue_nonbank, (1 - st.t.cdf(abs(T_test_nonbank[3]), len(log_annually)-2))*2)


Chi_test_nonbank = []
Chi_test_nonbank = np.append(Chi_test_nonbank, chi_square(log_monthly,22*log_daily.var()))
Chi_test_nonbank = np.append(Chi_test_nonbank, chi_square(log_quarterly,3*log_monthly.var()))
Chi_test_nonbank = np.append(Chi_test_nonbank, chi_square(log_annually,4*log_quarterly.var()))

#Chitest_pvalue_nonbank = []
#Chitest_pvalue_nonbank  = np.append(Chitest_pvalue_nonbank, (1 - st.t.cdf(abs(Chi_test_nonbank [0]), len(log_monthly)-2))*2)
#Chitest_pvalue_nonbank  = np.append(Chitest_pvalue_nonbank, (1 - st.t.cdf(abs(Chi_test_nonbank [1]), len(log_quarterly)-2))*2)
#Chitest_pvalue_nonbank  = np.append(Chitest_pvalue_nonbank, (1 - st.t.cdf(abs(Chi_test_nonbank [2]), len(log_annually)-2))*2)

skew_nonbank = []

skew_nonbank = np.append(skew_nonbank, skewness((log_daily-log_daily.mean())/log_daily.std()))
skew_nonbank = np.append(skew_nonbank, skewness((log_monthly-log_monthly.mean())/log_monthly.std()))
skew_nonbank = np.append(skew_nonbank, skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
skew_nonbank = np.append(skew_nonbank, skewness((log_annually-log_annually.mean())/log_annually.std()))

kurt_nonbank = []
kurt_nonbank = np.append(kurt_nonbank, kurtosis((log_daily-log_daily.mean())/log_daily.std()))
kurt_nonbank = np.append(kurt_nonbank, kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()))
kurt_nonbank = np.append(kurt_nonbank, kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
kurt_nonbank = np.append(kurt_nonbank, kurtosis((log_annually-log_annually.mean())/log_annually.std()))

JB_test_nonbank = [] 
JB_test_nonbank = np.append(JB_test_nonbank, jb_test(skewness((log_daily-log_daily.mean())/log_daily.std()),
                                                     kurtosis((log_daily-log_daily.mean())/log_daily.std()),
                                                     len(log_daily)))
JB_test_nonbank = np.append(JB_test_nonbank, jb_test(skewness((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     len(log_monthly)))
JB_test_nonbank = np.append(JB_test_nonbank, jb_test(skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     len(log_quarterly)))
JB_test_nonbank = np.append(JB_test_nonbank, jb_test(skewness((log_annually-log_annually.mean())/log_annually.std()),
                                                     kurtosis((log_annually-log_annually.mean())/log_annually.std()),
                                                     len(log_annually)))


#############     BANK STATISTIC        ########################################################  
log_daily, log_monthly, log_quarterly, log_annually = data_sampling(bank) 
T_test_bank = []
T_test_bank = np.append(T_test_bank, t_test(log_daily,0))
T_test_bank = np.append(T_test_bank, t_test(log_monthly, 22*log_daily.mean()))
T_test_bank = np.append(T_test_bank, t_test(log_quarterly, 3*log_monthly.mean()))
T_test_bank = np.append(T_test_bank, t_test(log_annually, 4*log_quarterly.mean()))

Ttest_pvalue_bank = []
Ttest_pvalue_bank  = np.append(Ttest_pvalue_bank, (1 - st.t.cdf(abs(T_test_bank[0]), len(log_daily)-2))*2)
Ttest_pvalue_bank  = np.append(Ttest_pvalue_bank, (1 - st.t.cdf(abs(T_test_bank[1]), len(log_monthly)-2))*2)
Ttest_pvalue_bank  = np.append(Ttest_pvalue_bank, (1 - st.t.cdf(abs(T_test_bank[2]), len(log_quarterly)-2))*2)
Ttest_pvalue_bank  = np.append(Ttest_pvalue_bank, (1 - st.t.cdf(abs(T_test_bank[3]), len(log_annually)-2))*2)


Chi_test_bank = []
Chi_test_bank = np.append(Chi_test_bank, chi_square(log_monthly,22*log_daily.var()))
Chi_test_bank = np.append(Chi_test_bank, chi_square(log_quarterly,3*log_monthly.var()))
Chi_test_bank = np.append(Chi_test_bank, chi_square(log_annually,4*log_quarterly.var()))

#Chitest_pvalue_bank = []
#Chitest_pvalue_bank   = np.append(Chitest_pvalue_bank , (1 - st.t.cdf(abs(Chi_test_bank [0]), len(log_monthly)-2))*2)
#Chitest_pvalue_bank   = np.append(Chitest_pvalue_bank , (1 - st.t.cdf(abs(Chi_test_bank [1]), len(log_quarterly)-2))*2)
#Chitest_pvalue_bank   = np.append(Chitest_pvalue_bank , (1 - st.t.cdf(abs(Chi_test_bank [2]), len(log_annually)-2))*2)

skew_bank = []
skew_bank = np.append(skew_bank, skewness((log_daily-log_daily.mean())/log_daily.std()))
skew_bank = np.append(skew_bank, skewness((log_monthly-log_monthly.mean())/log_monthly.std()))
skew_bank = np.append(skew_bank, skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
skew_bank = np.append(skew_bank, skewness((log_annually-log_annually.mean())/log_annually.std()))

kurt_bank = []
kurt_bank = np.append(kurt_bank, kurtosis((log_daily-log_daily.mean())/log_daily.std()))
kurt_bank = np.append(kurt_bank, kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()))
kurt_bank = np.append(kurt_bank, kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
kurt_bank = np.append(kurt_bank, kurtosis((log_annually-log_annually.mean())/log_annually.std()))


JB_test_bank = [] 
JB_test_bank = np.append(JB_test_bank, jb_test(skewness((log_daily-log_daily.mean())/log_daily.std()),
                                                     kurtosis((log_daily-log_daily.mean())/log_daily.std()),
                                                     len(log_daily)))
JB_test_bank = np.append(JB_test_bank, jb_test(skewness((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     len(log_monthly)))
JB_test_bank = np.append(JB_test_bank, jb_test(skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     len(log_quarterly)))
JB_test_bank = np.append(JB_test_bank, jb_test(skewness((log_annually-log_annually.mean())/log_annually.std()),
                                                     kurtosis((log_annually-log_annually.mean())/log_annually.std()),
                                                     len(log_annually)))
#############    INDEX STATISTIC        ########################################################  
log_daily, log_monthly, log_quarterly, log_annually = data_sampling(index) 
T_test_index = []
T_test_index = np.append(T_test_index, t_test(log_daily,0))
T_test_index = np.append(T_test_index, t_test(log_monthly, 22*log_daily.mean()))
T_test_index = np.append(T_test_index, t_test(log_quarterly, 3*log_monthly.mean()))
T_test_index = np.append(T_test_index, t_test(log_annually, 4*log_quarterly.mean()))

Ttest_pvalue_index = []
Ttest_pvalue_index  = np.append(Ttest_pvalue_index, (1 - st.t.cdf(abs(T_test_index[0]), len(log_daily)-2))*2)
Ttest_pvalue_index  = np.append(Ttest_pvalue_index, (1 - st.t.cdf(abs(T_test_index[1]), len(log_monthly)-2))*2)
Ttest_pvalue_index  = np.append(Ttest_pvalue_index, (1 - st.t.cdf(abs(T_test_index[2]), len(log_quarterly)-2))*2)
Ttest_pvalue_index  = np.append(Ttest_pvalue_index, (1 - st.t.cdf(abs(T_test_index[3]), len(log_annually)-2))*2)


Chi_test_index = []
Chi_test_index = np.append(Chi_test_index, chi_square(log_monthly,22*log_daily.var()))
Chi_test_index = np.append(Chi_test_index, chi_square(log_quarterly,3*log_monthly.var()))
Chi_test_index = np.append(Chi_test_index, chi_square(log_annually,4*log_quarterly.var()))

#Chitest_pvalue_index = []
#Chitest_pvalue_index   = np.append(Chitest_pvalue_index , (1 - st.t.cdf(abs(Chi_test_index [0]), len(log_monthly)-2))*2)
#Chitest_pvalue_index   = np.append(Chitest_pvalue_index , (1 - st.t.cdf(abs(Chi_test_index [1]), len(log_quarterly)-2))*2)
#Chitest_pvalue_index   = np.append(Chitest_pvalue_index , (1 - st.t.cdf(abs(Chi_test_index [2]), len(log_annually)-2))*2)
skew_index = []
skew_index = np.append(skew_index, skewness((log_daily-log_daily.mean())/log_daily.std()))
skew_index = np.append(skew_index, skewness((log_monthly-log_monthly.mean())/log_monthly.std()))
skew_index = np.append(skew_index, skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
skew_index = np.append(skew_index, skewness((log_annually-log_annually.mean())/log_annually.std()))

kurt_index = []
kurt_index = np.append(kurt_index, kurtosis((log_daily-log_daily.mean())/log_daily.std()))
kurt_index = np.append(kurt_index, kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()))
kurt_index = np.append(kurt_index, kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()))
kurt_index = np.append(kurt_index, kurtosis((log_annually-log_annually.mean())/log_annually.std()))



JB_test_index = [] 
JB_test_index = np.append(JB_test_index, jb_test(skewness((log_daily-log_daily.mean())/log_daily.std()),
                                                     kurtosis((log_daily-log_daily.mean())/log_daily.std()),
                                                     len(log_daily)))
JB_test_index = np.append(JB_test_index, jb_test(skewness((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     kurtosis((log_monthly-log_monthly.mean())/log_monthly.std()),
                                                     len(log_monthly)))
JB_test_index = np.append(JB_test_index, jb_test(skewness((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     kurtosis((log_quarterly-log_quarterly.mean())/log_quarterly.std()),
                                                     len(log_quarterly)))
JB_test_index = np.append(JB_test_index, jb_test(skewness((log_annually-log_annually.mean())/log_annually.std()),
                                                     kurtosis((log_annually-log_annually.mean())/log_annually.std()),
                                                     len(log_annually)))   
 
#############    combined STATISTIC        ########################################################      
    
total_t_test = pd.DataFrame([T_test_nonbank ,T_test_bank, T_test_index] , index=['SK Telecom Co Ltd','Industrial Bank of Korea','Korea Index'] ) 
total_t_test = total_t_test.transpose() 
total_t_test.index = ['Daily','Monthly','Quaterly','Annually']
print('t_test')
print(total_t_test)
print(' ')
total_Ttest_pvalue = pd.DataFrame([Ttest_pvalue_nonbank  ,Ttest_pvalue_bank , Ttest_pvalue_index] , index=['SK Telecom Co Ltd','Industrial Bank of Korea','Korea Index'] ) 
total_Ttest_pvalue = total_Ttest_pvalue.transpose()
total_Ttest_pvalue.index = ['P value Daily','P value Monthly','P value Quaterly','P value Annually']
print('p-value')
print(total_Ttest_pvalue)
print(' ')
total_chi_test = pd.DataFrame([Chi_test_nonbank ,Chi_test_bank, Chi_test_index] , index=['SK Telecom Co Ltd','Industrial Bank of Korea','Korea Index'] ) 
total_chi_test = total_chi_test.transpose()
total_chi_test.index = ['Monthly','Quaterly','Annually']
print('chi-test')
print(total_chi_test)
print(' ')
#total_chitest_pvalue = pd.DataFrame([Chitest_pvalue_nonbank, Chitest_pvalue_bank, Chitest_pvalue_index], index=['SK Telecom Co Ltd','Industrial Bank of Korea','Korea Index'] )
#total_chitest_pvalue = total_Ttest_pvalue.transpose( )
#total_chitest_pvalue.index = ['P value Monthly','P value Quaterly','P value Annually']

total_skew = pd.DataFrame([skew_bank, skew_nonbank, skew_index],  index = ['Bank','nonbank','Index'])
total_skew = total_skew.transpose()

total_kurt = pd.DataFrame([kurt_bank , kurt_nonbank , kurt_index ],  index = ['Bank','nonbank','Index'])
total_kurt = total_kurt.transpose()

total_JB_test = pd.DataFrame([JB_test_nonbank  , JB_test_bank , JB_test_index ] , index=['SK Telecom Co Ltd','Industrial Bank of Korea','Korea Index'] ) 
total_JB_test = total_JB_test.transpose()
total_JB_test.index = ['Daily','Monthly','Quaterly','Annually']
print('JB-test')
print(total_JB_test)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    