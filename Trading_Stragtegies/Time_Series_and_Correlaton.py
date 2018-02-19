# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:26:48 2018

@author: Jon Wee
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
from datetime import datetime
import time


### Apple stock analysis
end = datetime.now()
start = datetime(end.year-2, end.month, end.day)
CMT = web.DataReader("AAPL", 'yahoo', start, end)
# Figure 1
CMT['Adj Close'].plot(legend=True, figsize=(10, 5), \
                      title='Apple Inc.', \
                      label='Adjusted Closing Price')
# Figure 2
plt.figure(figsize=(10,5))
top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
top.plot(CMT.index, CMT['Adj Close']) #CMT.index gives the dates
bottom.bar(CMT.index, CMT['Volume'])
top.axes.get_xaxis().set_visible(False)
top.set_title('Apple Inc.')
top.set_ylabel('Adj Closing Price')
bottom.set_ylabel('Volume')
# Figure 3
plt.figure(figsize=(10,5))
sns.distplot(CMT['Adj Close'].dropna(), bins=50, color='purple')
# Figure 4
sma10 = CMT['Close'].rolling(10).mean() #10 days
sma20 = CMT['Close'].rolling(20).mean() #20 days
sma50 = CMT['Close'].rolling(50).mean() #50 days
CMTsma = pd.DataFrame({'CMT': CMT['Close'], 'SMA 10': sma10, 
                       'SMA 20': sma20, 'SMA 50': sma50})
CMTsma.plot(figsize=(10, 5), legend=True, title='Apple Inc.')
plt.show()


#%%  Finding correlation


end = datetime.now()
start = datetime(end.year-2, end.month, end.day)
AAPL = pd.read_csv("Yahoo_AAPL.csv", index_col=0, parse_dates=True)
AMZN = pd.read_csv("Yahoo_AMZN.csv", index_col=0, parse_dates=True)
HPQ = pd.read_csv("Yahoo_HPQ.csv", index_col=0, parse_dates=True)
INTC = pd.read_csv("Yahoo_INTC.csv", index_col=0, parse_dates=True)
MSFT = pd.read_csv("Yahoo_MSFT.csv", index_col=0, parse_dates=True)
T = pd.read_csv("Yahoo_T.csv", index_col=0, parse_dates=True)
# create new dataframe with just closing price for each stock
df = pd.DataFrame({'AAPL': AAPL['Adj Close'], 'AMZN': AMZN['Adj Close'],
                   'HPQ': HPQ['Adj Close'], 'INTC': INTC['Adj Close'],
                   'MSFT': MSFT['Adj Close'], 'T': T['Adj Close']})
# Figure 1: Plot Multiple Stocks
df.plot(figsize=(10,4))
plt.ylabel('Price')
# Figure 2a: Normalising Multiple Stocks
returnfstart = df.apply(lambda x: x / x[0])
returnfstart.plot(figsize=(10,4)).axhline(1, lw=1, color='black')
plt.ylabel('Return From Start Price')
# Figure 2b: Percentage Change
df2=df.pct_change()
df2.plot(figsize=(10,4))
plt.axhline(0, color='black', lw=1)
plt.ylabel('Daily Percentage Return')
# Figure 3: Correlation Plots
sns.jointplot('INTC', 'MSFT', df, kind='scatter', color='seagreen')
# Correlation
plt.figure(figsize=(8,8))
sns.linearmodels.corrplot(df.dropna())

# Figure 4: sns.PairGrid for correlation Plots
fig = sns.PairGrid(df.dropna()) 
# define top, bottom and diagonal plots
fig.map_upper(plt.scatter, color='purple')
fig.map_lower(sns.kdeplot, cmap='cool_d')
fig.map_diag(sns.distplot, bins=30)
plt.show()