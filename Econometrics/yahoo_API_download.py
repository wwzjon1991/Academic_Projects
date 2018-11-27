# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:07:45 2018

@author: Jon Wee
"""

import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
#%%
# download stock data from yahoo finance
# AAPL, AMZN, GOOG, JPM, FB, TWTR, GS, BABA, MSFT, MS
# after you dowload stock data to the local folder, you can comment this part

start = dt.datetime(2015,1,1)
end = dt.datetime(2018,1,1)

df_AAPL = pdr.get_data_yahoo('AAPL', start, end)
print(df_AAPL.head())
df_AAPL.to_csv('AAPL.csv', sep=',')

df_AMZN = pdr.get_data_yahoo('AMZN', start, end)
print(df_AMZN.head())
df_AMZN.to_csv('AMZN.csv', sep=',')

df_GOOG = pdr.get_data_yahoo('GOOG', start, end)
print(df_GOOG.head())
df_GOOG.to_csv('GOOG.csv', sep=',')

df_JPM = pdr.get_data_yahoo('JPM', start, end)
print(df_JPM.head())
df_JPM.to_csv('JPM.csv', sep=',')

df_FB = pdr.get_data_yahoo('FB', start, end)
print(df_FB.head())
df_FB.to_csv('FB.csv', sep=',')

df_TWTR = pdr.get_data_yahoo('TWTR', start, end)
print(df_TWTR.head())
df_TWTR.to_csv('TWTR.csv', sep=',')

df_GS = pdr.get_data_yahoo('GS', start, end)
print(df_GS.head())
df_GS.to_csv('GS.csv', sep=',')

df_BABA = pdr.get_data_yahoo('BABA', start, end)
print(df_BABA.head())
df_BABA.to_csv('BABA.csv', sep=',')

df_MSFT = pdr.get_data_yahoo('MSFT', start, end)
print(df_MSFT.head())
df_MSFT.to_csv('MSFT.csv', sep=',')

df_MS = pdr.get_data_yahoo('MS', start, end)
print(df_MS.head())
df_MS.to_csv('MS.csv', sep=',')

