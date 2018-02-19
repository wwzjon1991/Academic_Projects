# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:25:42 2018

@author: Jon Wee
"""

# Moving Average Crossover 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"]=[12,8] # (optional)
data=pd.read_csv('CC3.SI.csv',index_col=0)
data.drop(data.index[data['Volume']==0],inplace=True)
data['15d']= np.round(data['Close'].rolling(window=15).mean(),3)
data['50d']= np.round(data['Close'].rolling(window=50).mean(),3)
x=data['15d']-data['50d']
x[x>0]=1
x[x<=0]=0
y=x.diff()
idxSell=y[y<0].index
idxBuy=y[y>0].index
data['crossSell']=np.nan
data.loc[idxSell,'crossSell']=data.loc[idxSell,'Close']
data['crossBuy']=np.nan
data.loc[idxBuy,'crossBuy']=data.loc[idxBuy,'Close']
data[['Close', '15d', '50d','crossSell','crossBuy']].plot(
        style=['k-','b-','c-','ro','yo'],linewidth=1)
plt.title("Starhub 2015-2017")
plt.show()