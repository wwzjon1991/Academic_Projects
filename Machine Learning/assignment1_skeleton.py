import pandas as pd
import pandas_datareader as pdr
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%% Read data from iex

msft = pdr.DataReader('MSFT', 'iex', start = '2017-05-01', end = '2018-04-30')

msft = msft.rename(index = {i: pd.datetime(int(i[:4]), int(i[5:7]), int(i[8:10])) for i in msft.index.values.tolist()})

msft['seq'] = range(msft.shape[0])
#%%
x1 = np.column_stack((msft.shift(1)['close'],
                      msft.shift(2)['close'],
                      msft.shift(3)['close'],
                      msft.shift(4)['close'],
                      msft.shift(5)['close']))[5:]
y1 = msft['close'][5:]

train = np.array([True] * 227 + [False] * 20)

x1_train, x1_test, y1_train, y1_test = x1[train], x1[~train], y1[train], y1[~train]


def LR_RMSE(x_train, x_test, y_train, y_test):
    
    # Training Data
    M1training = linear_model.LinearRegression()
    M1training.fit(x_train, y_train)
    
    M1training_y_pred = M1training.predict(x_train)
    
    # Test Data
    M1test_y_pred = M1training.predict(x_test)

    # The root mean squared error
    RSME_train  = np.sqrt(mean_squared_error(y_train, M1training_y_pred))
    RSME_test   = np.sqrt(mean_squared_error(y_test, M1test_y_pred))

        
    return RSME_train, RSME_test


Qns1 = pd.DataFrame(index = ['RMSE on training', 'RMSE on test'], columns = ['Model 1', 'Model 2', 'Model 3'], dtype = np.float32)
# build your model 1 here
Qns1.iloc[0, 0], Qns1.iloc[1, 0] = LR_RMSE(x1_train, x1_test, y1_train, y1_test)


x2 = np.column_stack((x1, msft['seq'][5:]))
y2 = msft['close'][5:]
x2_train, x2_test, y2_train, y2_test = x2[train], x2[~train], y2[train], y2[~train]

# build your model 2 here
Qns1.iloc[0, 1], Qns1.iloc[1, 1] = LR_RMSE(x2_train, x2_test, y2_train, y2_test)


ma = msft.apply(lambda x: x.shift(1).rolling(5).mean()).dropna()['close']
x3 = np.column_stack((x2, ma))
y3 = msft['close'][5:]
x3_train, x3_test, y3_train, y3_test = x3[train], x3[~train], y3[train], y3[~train]

# build your model 3 here
Qns1.iloc[0, 2], Qns1.iloc[1, 2] = LR_RMSE(x3_train, x3_test, y3_train, y3_test)
print('Question 1: ')
print(Qns1)
print(" ")


#%%
diff = np.zeros((msft.shape[0] - 5, 5))
for i in range(5):
    diff[:,i] = msft.apply(lambda x: x.shift(i) - x.shift(i + 1))['close'][5:]


def my_discretize(m, step):
    n = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j] > 0.0:
                n[i,j] = np.ceil(m[i,j] / step)
            elif m[i,j] < 0.0:
                n[i,j] = -np.ceil(-m[i,j] / step)
    return n

# Question 2 answers
Qns2 = pd.DataFrame(index = ['RMSE on training', 'RMSE on test'], 
                    columns = ['Model step=1', 'Model step=0.5', 'Model step=0.2'],
                    dtype = np.float32)


for id, step in enumerate([1, 0.5, 0.2]):
    x = np.column_stack((x3, my_discretize(diff, step)))
    y = msft['close'][5:]
    xx_train, xx_test, yy_train, yy_test = x[train], x[~train], y[train], y[~train]
    # build your model here
    Qns2.iloc[0, id], Qns2.iloc[1, id] = LR_RMSE(xx_train, xx_test, yy_train, yy_test)

print('Question 2: ')
print(Qns2)
print(" ")
    

#%%
def RRmodel(x_train, x_test, y_train, y_test, A):
    
    # Training Data
    M1training = linear_model.Ridge(alpha = A)
    M1training.fit(x_train, y_train)
    
    M1training_y_pred = M1training.predict(x_train)
    
    # Test Data   
    M1test_y_pred = M1training.predict(x_test)
       
    # The root mean squared error
    RSME_train  = np.sqrt(mean_squared_error(y_train, M1training_y_pred))
    RSME_test   = np.sqrt(mean_squared_error(y_test, M1test_y_pred))

    # Model Complexity
    MComX = np.sqrt(np.sum(np.square(M1training.coef_)))
    
    return RSME_train, RSME_test, MComX

ind = ['step(1): RMSE train','step(1): RMSE test','step(1): Complexity',
       'step(0.5): RMSE train','step(0.5): RMSE test','step(0.5): Complexity',
       'step(0.2): RMSE train','step(0.2): RMSE test','step(0.2): Complexity']

Qns3 = pd.DataFrame(index = ind,
                    columns = ['Alpha=0.001','Alpha=0.01','Alpha=0.1','Alpha=1','Alpha=10'],
                    dtype = np.float32)
for index,step in enumerate([1, 0.5, 0.2]):
    x = np.column_stack((x3, my_discretize(diff, step)))
    y = msft['close'][5:]
    x_train, x_test, y_train, y_test = x[train], x[~train], y[train], y[~train]
    for col, AAlpha in enumerate([0.001, 0.01, 0.1, 1, 10]):
        # build your model here
        Qns3.iloc[index*3 , col],Qns3.iloc[index*3+1 , col],Qns3.iloc[index*3+2 , col] =\
                                                RRmodel(x_train, x_test, y_train, y_test, AAlpha)
    
print('Question 3: ')
print(Qns3)
print(" ")

# complexity decrease when alpha increase

#%% Plot graph
a = [0.001, 0.01, 0.1, 1, 10]
t = ['Step = 1','Step = 0.5','Step = 0.2']

for step, id in enumerate(range(0,9,3)):
    fig, ax1 = plt.subplots(figsize = (10, 6))

    ax1.plot(a, Qns3.iloc[id, :].values, 'go--', label = str(Qns3.index[id]))
    ax1.plot(a, Qns3.iloc[id+1, :].values, 'bo--', label = str(Qns3.index[id+1]))
    
    ax2 = ax1.twinx()
    
    ax2.plot(a, Qns3.iloc[id+2, :].values, 'ro--',
             label = str(Qns3.index[id+2]))

    ax1.set_ylabel('RSME', fontsize=12)
    ax1.set_xlabel('Alpha', fontsize=12)
    ax2.set_ylabel('Model Complexity', fontsize=12)
    
        
    ax1.legend(loc='upper center', bbox_to_anchor=(0.1,1.15))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.9,1.15))
    plt.title('Ridge Regression Model ' + t[step], fontsize = 12)
    plt.show()



