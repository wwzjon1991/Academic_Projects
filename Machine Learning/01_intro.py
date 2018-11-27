import pandas as pd
import numpy as np

d0104 = pd.read_csv('20160104.csv')
d0104.columns
d0104.dtypes

d0104 = pd.read_csv('20160104.csv',
                    dtype = {'SIZE': np.uint32,
                             'PRICE': np.float32, # float 32 save date memory usage
                             'NBO': np.float32,
                             'NBB': np.float32},
                    parse_dates = {'datetime': ['DATE', 'TIME_M']},
                    date_parser = lambda x, y: pd.datetime.strptime(x + ':' + y[:y.rfind('.')+7], '%d%b%Y:%H:%M:%S.%f'))
                    # +7 is get 6 decimals only
#%%
d0104.datetime
                    
#%% find the number of rows
d0104.shape

#%% show the top 5 rows
d0104.head()
#%% show the bottom 3 rows
d0104.tail(3)

#%% statistic of the columns in numericals
d0104.describe()

#%% find the freq values

d0104.SYM_ROOT.value_counts()
#%%

d0104.type.value_counts()
#%%

d0104.EX.value_counts()
#%%

d0104[0:3]
#%%

d0104.loc[0:3]
#%% inclusive of lower bound but exclusive of upper bound

d0104.iloc[0:3]
#%% doesn't work for nesting
"""
d0104[0:3, ['PRICE', 'NBO', 'NBB']]
"""
#%%  NBO best off | NBB best offer

d0104.loc[0:3, ['PRICE', 'NBO', 'NBB']]

#%% only for intergers

d0104.iloc[0:3, 5:8]
#%%

df = d0104[(d0104.PRICE <= d0104.NBO) & (d0104.PRICE >= d0104.NBB)]
df.iloc[0:3, 5:8]

#%%
df.shape

#%% it is different?? why?
d0104.shape

#%%

import matplotlib.pyplot as plt
#%%

df[df.SYM_ROOT == 'MMM'].plot('datetime', 'PRICE')
#%%

df4 = df[df.SYM_ROOT.isin(['CSCO', 'HPQ', 'IBM', 'INTC'])]
#%%

df4.set_index(df4.datetime, drop = False, inplace = True)
df4.groupby(df4.SYM_ROOT)['PRICE'].plot(legend = True, grid = True)
plt.show()

#%%
grp = df4.groupby(df4.SYM_ROOT) # group the tickers
fig, axes = plt.subplots(2, 2, figsize=(12,12))
plt.subplots_adjust(hspace=0.8)
for k, a in zip(grp.groups.keys(), axes.flatten()):
    grp.get_group(k).plot('datetime', 'PRICE', ax = a, title = k, legend = False)
plt.show()

#%%
import pandas_datareader as pdr 

msft = pdr.DataReader('MSFT', 'iex', start = '2017-05-01', end = '2018-04-30')
aapl = pdr.DataReader('AAPL', 'iex', start = '2017-05-01', end = '2018-04-30')
goog = pdr.DataReader('GOOG', 'iex', start = '2017-05-01', end = '2018-04-30')

#%% non-normalise graph
# the date is string, the x-axis is not date
# need to convert to datetime format

mag = pd.DataFrame({'MSFT': msft['close'], 'AAPL': aapl['close'],\
                    'GOOG': goog['close']})
mag = mag.rename(index = {i: pd.datetime(int(i[:4]), int(i[5:7]), \
                         int(i[8:10])) for i in goog.index.values.tolist()})
mag.plot(grid = True)

#%%
mag.plot(secondary_y = 'GOOG', grid = True)

#%% standardise the first data the stock movement

mag.apply(lambda x: x / x[0]).plot(grid = True)

#%% Returns
mag.apply(lambda x: (x.shift(1) - x) / x).plot(grid = True)# shift(1) to shift on the next data

#%%
fig, axes = plt.subplots(2, 2, figsize=(12,12))
plt.subplots_adjust(wspace = 0.4)
for i, a in zip(range(5, 21, 5), axes.flatten()):
    mag.apply(lambda x: x.rolling(i).mean()).plot(secondary_y = 'GOOG',\
             grid = True, ax = a, title = 'moving window size: ' + str(i))
plt.show()
