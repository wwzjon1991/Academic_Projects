# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:21:36 2018

@author: Jon Wee
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy.stats import pearsonr, zscore, f
import seaborn as sns
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import quandl
import json

#%% Function
# define function to test for factors
def test(factor_series, result, df_combine):

    # combine into a dataframe for testing
    df_test = pd.DataFrame(index=df_combine.index)
    df_test[factor_series] = df_combine[factor_series]
    df_test['returns'] = df_combine['BTC_Returns']
    df_test = df_test.astype(float)


#    print(df_test.head())
    

    # Regression test
    if result == "regression":
        
        print(factor_series)
        reg_results = sm.OLS(df_test['returns'].astype(float), \
                             sm.add_constant(df_test[factor_series].astype(float))).fit()
        print(reg_results.summary())
        plt.rc('figure', figsize=(15, 8))
        plt.text(0.01,0.01,str(reg_results.summary()),{'fontsize': 3}, fontproperties = 'monospace')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(factor_series +'regression.png')
        plt.show()
        
    # Granger causality test
    elif result == "granger":
        
        print(factor_series)
        gc_results = grangercausalitytests( df_test, maxlag=4, addconst=True, verbose=True)
        optimal_lag = -1
        F_test = -1.0
        for key in gc_results.keys():
            _F_test_ = gc_results[key][0]['params_ftest'][0]
            if _F_test_ > F_test:
                F_test = _F_test_
                optimal_lag = key
        return optimal_lag,gc_results
        
    # Pearson correlation test
    elif result == "pearson":
        print(factor_series)
        pc_results = pearsonr(df_test['returns'].astype(float), df_test[factor_series].astype(float))
        print("Pearson's Correlation Coefficient, p-value: " + str(pc_results))
        return pc_results
    
    # For faster results  
    elif result == "all":
        print(factor_series)
        reg_results = sm.OLS(df_test['returns'].astype(float), \
                             sm.add_constant(df_test[factor_series].astype(float))).fit()
        print(reg_results.summary())
        plt.rc('figure', figsize=(16, 9))
        plt.text(0.01,0.01,str(reg_results.summary()),{'fontsize': 3}, fontproperties = 'monospace')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig( factor_series +'regression.png')
        plt.show()

        print(factor_series)
        gc_results = grangercausalitytests(df_test, maxlag=8, addconst=True, verbose=True)

        
        print(factor_series)
        pc_results = pearsonr(df_test['returns'].astype(float), df_test[factor_series].astype(float))
        print("Pearson's Correlation Coefficient, p-value: " + str(pc_results))

#%% Do Twitter Sentimental Analysis - Vader
#df_sa = pd.DataFrame(index=['Sentiment', 'Price'], columns=range(1, 16))
#df_full = pd.DataFrame()
#
## Twitter Account Users
#users = ['aantonop','BitcoinNetworks','BitcoinMagazine','BTCTN','BTCnewsBOT','BTCNewsletter','Coindesk','Cointelegraph',\
#         'ForbesCrypto','RedditBTC','NeerajKA','VitalikButerin','SatoshiLite','WhalePanda','NickSzabo4','gavinandresen',\
#         'brian_armstrong','starkness','twobitidiot','lopp','rogerkver','Excellion','ErikVoorhees','TuurDemeester']
#
## import excel twitter output spreadsheets
#for i in users:
#    xl_output = pd.read_excel(str(i) + '.xlsx')
#
#    # drop columns which are not needed
#    xl_output = xl_output.drop(columns=['fullname', 'id', 'likes', 'replies', 'retweets', 'user'])
#    xl_output = xl_output.replace(0, np.nan)
#    xl_output = xl_output.dropna(axis=0, how='any')
#    xl_output['sentiment'] = np.nan
#
#    # conduct Vader sentiment analysis on data
#    analyzer = SentimentIntensityAnalyzer()
#    for j in xl_output.index:
#        sentence = xl_output.loc[j]['text']
#        xl_output.at[j, 'sentiment'] = analyzer.polarity_scores(sentence)['compound']
#
#    # store monthly sentiment scores
#    df_sa.loc['Sentiment'][i] = xl_output['sentiment'].mean()
#
#    # full daily data
#    df_full = pd.concat([df_full, xl_output])
#    
## adjust data
#df_full = df_full.sort_values('timestamp')
#df_full = df_full.set_index('timestamp')
#df_full2 = df_full.resample('W', how='mean')
#df_full2.index = df_full2.index.astype(str).str.slice(0, 10)

#%%
# pull data from quandl package with date range & weekly
#df_btc = pd.read_csv('btc_price.csv')
#df_btc.index = df_full2.index
#
#
## Create dataframe to consildate reseults
#df_tweet = pd.DataFrame(columns = ['BTC_Returns', 'sentiment'])
#df_tweet['sentiment'] = df_full2['sentiment'].pct_change(1)
#df_tweet['BTC_Returns'] = df_btc['Last'].pct_change(1)
#
##df_tweet['sentiment'] = np.log(df_full2['sentiment']/df_full2['sentiment'].shift(1))
##df_tweet['BTC Returns'] = np.log(df_btc['Last']/df_btc['Last'].shift(1))
#
#df_tweet.dropna(inplace = True)
#df_tweet = df_tweet.astype(float)
#
#
#for col in df_tweet.columns[1:]:
#    test(col, 'all', df_tweet)        
        
#%% Import Significant Results        
        
# Sample Codes for Research Methods #
# Group 7

# Conslidated the data - BTC last price, Google Trends, PageViews, BTC Address 

# pull data from excel
df_comb = pd.read_excel('Project Data.xlsx')
df_comb = df_comb.set_index('Date')
# Scaling the data factor to prevent multicollinearity errors
#df_comb['BTC Add'] = df_comb['BTC Add'] / 10000
# Scaling the data factor to prevent multicollinearity errors
#df_comb['CoinBase'] = df_comb['CoinBase'] * 10000

# Using Change percent
df_comb = df_comb.pct_change(1)

df_comb.columns = ['BTC_Returns', 'BTC_Address','Google','PageViews']
df_comb = df_comb[['BTC_Returns', 'Google','PageViews','BTC_Address']] #rearragning order for heatmap

df_comb.dropna(inplace = True)

     
#%% plot Time-Series        

# Plot the Time Series
df_test = df_comb.copy()
# Normalising the Timeseries
# df_test = np.log(df_test / df_test.shift(1))
#for col in df_test.columns:
#    df_test[col] = zscore(df_test[col])
df_test = df_test.astype(float)
  
print("Time Series")
colours1 = ['black','xkcd:faded green', 'xkcd:faded blue', 'xkcd:faded red']

fig = plt.figure(figsize=(28,15))

for i, x in enumerate(df_test.columns[1:]):
    ax=plt.subplot(3,1,i+1)

    plt.plot(df_test.index, df_test['BTC_Returns'],color= colours1[0], marker='o',\
             linestyle='dashed', label = 'BTC_Returns' )
    
    plt.plot(df_test.index, df_test[x],color= colours1[i+1], marker='o', \
             linestyle='dashed', label = x )
    plt.grid(True)
    plt.legend()
    plt.ylabel(x, fontsize= 20)
    plt.xlabel('Date')

plt.savefig('timeseries.png')    
plt.show()    

#%% Dicky Fuller 

# To test if time-series is stationary 
station = df_comb.copy()

df_dickey = pd.DataFrame(index = ['ADF Statistic','P-Value','Critical Values 1%','Critical Values 5%', \
                                  'Critical Values 10%'] ,columns =station.columns )

for col,i in enumerate(station.columns):
#    print('Dickey Fuller Test:', i)
    result = adfuller(station[i])
#    print('ADF Statistic: %f' % result[0])
    df_dickey.iloc[0, col] = result[0]
#    print('p-value: %f' % result[1])
    df_dickey.iloc[1, col] = result[0]
#    print('Critical Values:')
    idd = 0 
    for key, value in result[4].items():
#        print('\t%s: %.3f' % (key, value))
        df_dickey.iloc[2+idd, col] = value
        idd +=1

df_dickey = df_dickey.astype('float')        
with open('df_dickey.tex','w') as tf:
    tf.write(df_dickey.to_latex())      
           
#%%  Regression
# test each factor
for i in df_comb.columns[1:]:
    test(i, 'regression', df_comb)        
        
       

#%% Regression Graph 
colours = ['r','b','g', 'k' ]
sns.set(color_codes=True, font_scale=1.5)
for id,x in enumerate(df_comb.columns[1:]):
    
    plt.subplots(figsize=(9, 5))
    sns.regplot(df_comb[x],df_comb['BTC_Returns'], color=colours[id])
#    plt.title(x , fontsize= 20)
    plt.savefig(x+'plot.png')
    plt.show()        
        
        
#%%  # Granger Causality Test

df_granger = pd.DataFrame(index = ['Optimal lag'],columns = df_comb.columns[1:])    

df_gstat = pd.DataFrame(columns = df_granger.columns)
df_gstat.loc['statistic', :] = [['F-critical', 'F-test', 'P-value']]
# test each factor
for id,i in enumerate(df_granger.columns):
    print(id,i)
    df_granger.loc['Optimal lag', i], gstat = test(df_comb.columns[id+1], 'granger', df_comb)   

    for key in gstat.keys():
            denom, num = gstat[key][0]['params_ftest'][2] , gstat[key][0]['params_ftest'][3]
            df_gstat.loc['lag='+str(key), i ] = np.array([f.pdf(0.9, dfn=num, dfd=denom),\
                                                                 gstat[key][0]['params_ftest'][0],\
                                                                 gstat[key][0]['params_ftest'][1]],dtype = np.float32)

    
    
with open('df_gstat.tex','w') as tf:
    tf.write(df_gstat.to_latex())              
print(df_gstat)

#%% Pearson Correlation
# test each factor
df_pearson = pd.DataFrame(columns = df_comb.columns)
for i in df_comb.columns[1:]:
    df_pearson.loc['Coeff' , i],df_pearson.loc['P-value' , i] = test(i, 'pearson', df_comb)    

df_pearson = df_pearson.iloc[:, 1:]
df_pearson = df_pearson.astype(float)
        
with open('df_pearson.tex','w') as tf:
    tf.write(df_pearson.to_latex())  

#%% Correlation table
corr = df_comb.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap

cmap = sns.diverging_palette(1690, 10, sep=20, as_cmap=True)
# Plot
sns.heatmap(corr, cmap=cmap)
plt.title('Correlation Matrix', fontsize= 20)
plt.savefig('pearson.png')
plt.show()        

#%% Testing for Optimal Granger

#df_optimal = df_comb.copy()
#df_optimal.columns = ['BTC_Returns', 'OptGoogle','OptPageViews','OptBTC_Address']
#
#for id,lags in enumerate(df_granger.iloc[0]):
#    col = df_optimal.columns[id+1] 
#    print(col, lags)
#    df_optreg = pd.DataFrame()
#    df_optimal[col] = df_optimal[col].shift(lags)  
#    df_optreg = df_optimal.dropna()
#
#    test(col, 'regression', df_optreg)   

#    plt.subplots(figsize=(9, 9))
#    sns.regplot(df_optimal['OptPageViews'],df_optimal['BTC Returns'], color=colours[id])
#    plt.title(x , fontsize= 20)
#    plt.show()  
#       
#%% Multi-Factor Regression

df_multi = df_comb.copy()
df_multi = df_multi.astype(float)

reg_results = sm.OLS(df_multi['BTC_Returns'].astype(float), sm.add_constant(df_multi.iloc[:, 1:].astype(float))).fit()
print(reg_results.summary())
plt.rc('figure', figsize=(16, 9))
plt.text(0.01,0.01,str(reg_results.summary()),{'fontsize': 3}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('mulregression.png')
plt.show()      
        
        
        
        
        
        
        