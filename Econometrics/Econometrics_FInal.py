#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:25:33 2018

@author: charlotteliu
"""
from __future__ import division, print_function
import fix_yahoo_finance as fyf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

namelist=pd.read_excel('Summary_All.xlsx',sheet_name='Group 2')

#%% download data
#for i in range(len(namelist['Ticker'])):
#    ticker = namelist['Ticker'][i]
#    data = fyf.download(ticker, start="1980-01-01", actions=True)
#    out_filename = ticker +  ".csv"
#    data.to_csv(out_filename)

#ticker = 'SPY'
#data = fyf.download(ticker, start="1980-01-01", actions=True)
#out_filename = ticker +  ".csv"
#data.to_csv(out_filename)

#%%
# get time series for one ticker
def read_GFD(ticker):
   fn = ticker + ".csv"
   data = pd.read_csv(fn)
   dates = pd.to_datetime(data['Date'], format='%Y/%m/%d')
   price = data['Close']
   ts = pd.Series(price)
   ts.index = dates
   return ts

def event_lr(stock, edate, L, W):
# edate is the announcement date
   n = len(stock)
   for i in range(n):
      if stock.index[i] == edate:
         break
     
   ei1 = i - (L + W + 1)
   ei2 = i + W + 1
   if ei1 < 0:
       ei1 = 0
   sa = stock[ei1:ei2]
   sb = sa.pct_change()
   sc = sb.dropna()
   sd = np.log(1+sc)
   return sd, sa

def ols(x, y):
   xbar = np.mean(x)
   ybar = np.mean(y)
   
   dx = x - xbar
   dy = y - ybar

   dx2 = sum(dx*dx)

   bhat = sum(dx*dy)/dx2
   ahat = ybar - bhat*xbar

   n = len(x)
   residual = y - ahat - bhat*x
   sigma2 = sum(residual*residual)/(n-2)
   
   return ahat, bhat, sigma2, xbar, dx2
#%%
####################################################################################
AllAR = pd.DataFrame()

for i in range(len(namelist['Ticker'])):
    stock = read_GFD(namelist['Ticker'][i])
    etf = read_GFD('SPY')

    L, W =  240, 10

    edate = namelist['Announcement'][i]
    
    while(edate not in stock.index):
        edate = edate + datetime.timedelta(days=1)
    edate = pd.to_datetime(edate)            

# calculate the time series that we use later
    lrstock, ss = event_lr(stock, edate, L, W)
    lretf, es = event_lr(etf, edate, L, W)

# linear regression of historical data, CAPM model 
    ahat, bhat, sigma2, rmbar, drm2sum = ols(lretf[0:L].values, lrstock[0:L].values)

# abormal return
    ar = lrstock[L:] - (ahat + bhat*lretf[L:])
    ar.name = 'AR'

# market return - average market return, a numpy array
    drm = lretf[L:] - rmbar

# variance of abnormal return for each tao in the event window, a numpy array
    s2 = sigma2*(1 + 1/L + drm*drm/drm2sum)

# cumsum of the AR list = CAR, a numpy array
    car = ar.cumsum()
    car.name = 'CAR'

# AR array of the ith event
    AllAR[i] = np.array(ar)

# CAR variance & t-static, a numpy array
    carv = s2.cumsum()
    car_tstat = car/np.sqrt(carv)
    car_tstat.name = 'CAR t Score'

# AR variance & t-static
    ar_tstat = ar/np.sqrt(s2)
    ar_tstat.name = 'AR t Score'

####################################################################################
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 20}

    plt.rc('font', **font)

# plot CAR
    fig, ax1 = plt.subplots(figsize=(8,6))
    x = range(-W,W+1)
    ax1.plot(x, car, '-ro')
    ax1.grid(color = 'r', linestyle='dotted', lw=0.5)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(namelist['Ticker'][i] + ' CAR')
    legend = ax1.legend(loc='lower left', shadow=False)
    ax1.set_ylabel('Cumulative Abnormal Return')
    
# plot CAR t-static
    ax2 = ax1.twinx()
    ax2.plot(x, car_tstat, '-bx')
    ax2.set_ylabel('$t$ Score')
    ax2.grid(color = 'b', linestyle='dashed', lw=0.5)

    plt.xlabel('')
    fig.autofmt_xdate()
    legend = ax2.legend(loc='upper right', shadow=False)
    #plt.savefig('es_CAR.png', format = 'png', dpi=3*96)
    plt.show()

# plot AR
    fig, ax3 = plt.subplots(figsize=(8,6))
    ax3.plot(x, ar, '-ro')
    ax3.grid(color = 'r', linestyle='dotted', lw=0.5)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(namelist['Ticker'][i] + ' AR')
    legend = ax3.legend(loc='lower left', shadow=False)
    ax3.set_ylabel('Abnormal Return')
    
# plot AR t-static    
    ax4 = ax3.twinx()
    ax4.plot(x, ar_tstat, '-bx')
    ax4.set_ylabel('$t$ Score')
    ax4.grid(color = 'b', linestyle='dashed', lw=0.5)
    
    plt.xlabel('')
    fig.autofmt_xdate()
    legend = ax4.legend(loc='upper right', shadow=False)
    #plt.savefig('es_CAR.png', format = 'png', dpi=3*96)
    plt.show()
    
#%%
AllAR = AllAR.T

aar = pd.DataFrame.mean(AllAR)
aar.name = 'AAR'

caar = aar.cumsum()
caar.name = 'CAAR'

aar_tstat = aar/pd.DataFrame.std(aar)
aar_tstat.name = 'AAR t Score'

caar_tstat = caar/pd.DataFrame.std(caar)
caar_tstat.name = 'CAAR t Score'


# plot AAR   
fig, ax5 = plt.subplots(figsize=(8,6))
x = range(-W,W+1)
ax5.plot(x, aar, '-ro')
ax5.grid(color = 'r', linestyle='dotted', lw=0.5)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('AAR')
legend = ax5.legend(loc='lower left', shadow=False)
ax5.set_ylabel('Average Abnormal Return')

# plot AAR t-static    
ax6 = ax5.twinx()
ax6.plot(x, aar_tstat, '-bx')
ax6.set_ylabel('$t$ Score')
ax6.grid(color = 'b', linestyle='dashed', lw=0.5)
plt.xlabel('')
fig.autofmt_xdate()
legend = ax6.legend(loc='upper right', shadow=False)
plt.show()

# plot CAAR    
fig, ax7 = plt.subplots(figsize=(8,6))
ax7.plot(x, caar, '-ro')
ax7.grid(color = 'r', linestyle='dotted', lw=0.5)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('CAAR')
legend = ax7.legend(loc='lower left', shadow=False)
ax7.set_ylabel('Cumulative Average Abnormal Return')

# plot CAAR t-static    
ax8 = ax7.twinx()
ax8.plot(x, caar_tstat, '-bx')
ax8.set_ylabel('$t$ Score')
ax8.grid(color = 'b', linestyle='dashed', lw=0.5)
plt.xlabel('')
fig.autofmt_xdate()
legend = ax8.legend(loc='upper right', shadow=False)
plt.show()