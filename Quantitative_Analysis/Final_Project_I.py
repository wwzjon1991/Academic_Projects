# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 01:29:22 2017

@author: Jon Wee
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Excel files need to be in same folder/directionary
index = 'KOSPI--Index.csv'
bank = '024110-KS-Equity.csv'
non_bank = '017670-KS-Equity.csv'

#############PART ONE###############################################################
def data_sampling(X, Y):
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
    log_monthly = np.log(monthly) - np.log(monthly.shift(1))
    log_quarterly = np.log(quarterly) - np.log(quarterly.shift(1))
    log_annually = np.log(annually) - np.log(annually.shift(1))
    
########   droping first row of the log ret   ###########################################    
    log_daily = log_daily[1:]
    log_monthly = log_monthly[1:]
    log_quarterly = log_quarterly[1:]
    log_annually = log_annually[1:]
    
#    return log_daily, log_monthly, log_quarterly, log_annually
#log_daily, log_monthly, log_quarterly, log_annually = data_sampling(non_bank) 
#    return log_daily, log_monthly, log_quarterly, log_annually
#log_daily, log_monthly, log_quarterly, log_annually = data_sampling(non_bank)   
####     mean returns       #########################################################    
    mean_daily = np.mean(log_daily)*(len(log_daily)/17)
    mean_monthly = np.mean(log_monthly)*12
    mean_quarterly = np.mean(log_quarterly)*4 
    mean_annually = np.mean(log_annually)
    mean_total = pd.DataFrame(data=[mean_daily,mean_monthly, mean_quarterly, mean_annually], columns=[Y])
#    return mean_daily, mean_monthly, mean_quarterly, mean_annually
#mean_daily, mean_monthly, mean_quarterly, mean_annually = data_sampling(bank) 
####     std dev returns     ######################################################## 
    std_daily = np.std(log_daily,ddof=1)*((len(log_daily)/17)**0.5)
    std_monthly = np.std(log_monthly,ddof=1)*(12**0.5) 
    std_quarterly = np.std(log_quarterly,ddof=1)*(4**0.5)
    std_annually = np.std(log_annually,ddof=1)
    std_total = pd.DataFrame(data=[std_daily, std_monthly, std_quarterly, std_annually], columns=[Y])
#    return std_daily, std_monthly, std_quarterly, std_annually
#std_daily, std_monthly, std_quarterly, std_annually = data_sampling(index) 
####     PERCENTILES     ############################################################
    PER = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    PER_INDEX = ['Minimum', '1st Percentile', '5th Percentile', '10th Percentile', '25th Percentile','50th Percentile','75th Percentile','90th Percentile','95th Percentile','99th Percentile','Maximum']
    Daily_Per = log_daily.quantile(PER)
    Monthly_Per = log_monthly.quantile(PER)
    Quarterly_Per = log_quarterly.quantile(PER)
    Annual_Per = log_annually.quantile(PER)
    
    #Per_index = ['minimum', '1st percentile', '5th percentile', '10th percentile', '25th percentile', '50th percentile', '75th percentile', '90th percentile', '95th percentile', '99th percentile', 'maximum']
    percentiles = pd.DataFrame({'Daily': Daily_Per, 'Monthly': Monthly_Per, 'Quarterly': Quarterly_Per, 'Annual': Annual_Per}, index=PER)    
    percentiles.index = PER_INDEX 
#    return percentiles
#percentiles = data_sampling(non_bank) 

    
####     PLOT TIME SERIES   #########################################################    
    plt.figure(figsize=(8 , 6))
    plt.plot(data['PX_LAST'])
    plt.title('Time Series: %s' %Y)
    plt.xlabel('time')
    plt.ylabel('%s price' %Y)
    plt.show()
#data_sampling(bank, 'Industrial Bank of Korea')   

####     PLOT HISTGRAM     ##########################################################    
    #Daily 
    plt.figure(figsize=(8 , 6))
    plt.hist(log_daily, 60)
    plt.title('Daily Returns: %s' %Y)
    plt.xlabel('log return')
    plt.ylabel('frequency')
    plt.show()  
    #monthly 
    plt.figure(figsize=(8 , 6))
    plt.hist(log_monthly, 45)
    plt.title('Monthly Returns: %s' %Y)
    plt.xlabel('log return')
    plt.ylabel('frequency')
    plt.show() 
    #quarterly 
    plt.figure(figsize=(8 , 6))
    plt.hist(log_quarterly, 20)
    plt.title('Quarterly Returns: %s' %Y)
    plt.xlabel('log return')
    plt.ylabel('frequency')
    plt.show() 
    #annually 
    plt.figure(figsize=(8 , 6))
    plt.hist(log_annually, 8)
    plt.title('Annually Returns: %s' %Y)
    plt.xlabel('log return')
    plt.ylabel('frequency')
    plt.show() 
    
    return mean_total, std_total, daily, percentiles

"""    
#####################################################################################
####    CANT FIGURE OUT HOW TO PLOT THE P.D.F ON HISTOGRAM    #######################
#####################################################################################
"""    
    
mean_bank, std_bank, daily_bank, per_bank = data_sampling(bank, 'Industrial Bank of Korea')     
mean_nonbank, std_nonbank, daily_nonbank, per_nonbank = data_sampling(non_bank, 'SK Telecom Co Ltd')     
mean_index, std_index, daily_index, per_index = data_sampling(index, 'Korea Index')     

ret_bank = daily_bank.pct_change()
ret_bank = ret_bank[1:]
ret_bank = (1+ret_bank).cumprod() 

ret_nonbank = daily_nonbank.pct_change()
ret_nonbank = ret_nonbank[1:]
ret_nonbank = (1+ret_nonbank).cumprod() 

ret_index = daily_index.pct_change()
ret_index = ret_index[1:]
ret_index = (1+ret_index).cumprod() 

plt.figure(figsize=(8 , 6))
plt.plot(ret_bank, 'b', label='Industrial Bank of Korea')
plt.plot(ret_index, 'r--', label='Korea Index')    
plt.title('Cumulative Return Industrial Bank of Korea' )
plt.xlabel('Time')
plt.ylabel('Return %')
plt.legend()

plt.figure(figsize=(8 , 6))
plt.plot(ret_nonbank, 'g', label='SK Telecom Co Ltd')
plt.plot(ret_index, 'r--', label='Korea Index')    
plt.title('Cumulative Return SK Telecom Co Ltd' )
plt.xlabel('Time')
plt.ylabel('Return %')
plt.legend()

plt.figure(figsize=(8 , 6))
plt.plot(ret_bank, 'b', label='Industrial Bank of Korea')
plt.plot(ret_nonbank, 'g', label='SK Telecom Co Ltd')
plt.plot(ret_index, 'r--', label='Korea Index')    
plt.title('Overall Cumulative Return' )
plt.xlabel('Time')
plt.ylabel('Return %')
plt.legend()

  
Mean_overall = pd.concat([mean_bank,mean_nonbank,mean_index], axis=1) 
Mean_overall.index = ['daily', 'monthly','quarterly','annually']   
Std_overall =  pd.concat([std_bank, std_nonbank,std_index], axis=1)  
Std_overall.index = ['daily', 'monthly','quarterly','annually']        

print("Mean return")
print(Mean_overall)
print("Standard Deviation")
print(Std_overall)  
print("Industrial Bank of Korea Percentitles of Log Returns")  
print(per_bank) 
print("SK Telecom Co Ltd Percentitles of Log Returns")  
print(per_nonbank) 
print("KOSPI index Percentitles of Log Returns")  
print(per_index) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    