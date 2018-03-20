import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.stats import norm
from scipy.integrate import quad
from math import log, exp, sqrt
from scipy import interpolate

df_OIS = pd.read_excel('IR Data.xlsx', 'OIS', parse_cols=[0,2])
df_LIBOR = pd.read_excel('IR Data.xlsx', 'IRS', parse_cols=[0,2])

LIBOR_6m = df_LIBOR['Rate'][0]

for id,i in enumerate(df_OIS['Tenor']):
    if i[-1] == 'm' :
        df_OIS.loc[id, 'Tenor'] = np.float(i[:-1])/12
        df_LIBOR.loc[id, 'Tenor'] = np.float(i[:-1])/12
    elif i[-1] == 'y' :
        df_OIS.loc[id, 'Tenor'] = np.float(i[:-1])
        df_LIBOR.loc[id, 'Tenor'] = np.float(i[:-1])
        
Time = np.arange(0.5,30.5,0.5)
df_comb = pd.DataFrame(data=Time, columns=['Tenor'])
df_comb = df_comb.assign(OIS_DF=np.nan, LIBOR_DF=np.nan,F_LIBOR=np.nan)
###############################################################################
#%% Part I (Bootstrapping Swap Curves)
N = len(df_comb)
for id in range(N):
    for i in range(len(df_OIS)):
        if df_OIS.loc[i, 'Tenor'] == df_comb.loc[id, 'Tenor'] :           
            if df_LIBOR.loc[i, 'Tenor'] == 0.5:
                df_comb.loc[id, 'LIBOR_DF'] = 1/(1+0.5*df_LIBOR.loc[i, 'Rate'])
                df_comb.loc[id, 'IRS'] = df_LIBOR.loc[i, 'Rate'] 
                df_comb.loc[id, 'F_LIBOR'] = df_LIBOR.loc[i, 'Rate'] 
            elif df_LIBOR.loc[i, 'Tenor']>0.5:
                df_comb.loc[id, 'IRS'] = df_LIBOR.loc[i, 'Rate']               
            df_comb.loc[id, 'OIS_DF'] = 1/(1+df_OIS.loc[i, 'Tenor']*df_OIS.loc[i, 'Rate'])
        
df_comb['OIS_DF'].interpolate(method='linear',inplace=True)

IDX = df_comb.index[np.invert(np.isnan(np.array(df_comb['IRS'])))]
index1 = IDX[2:] # end
index2 = IDX[1:-1] # start

def Interp(start,end,mid,x):# 1.5,2.5,3.5,4.5=id, x=2y,3y,4y,5y
    Df_mid = df_comb.loc[start, 'LIBOR_DF']+(x-df_comb.loc[start, 'LIBOR_DF'])/ \
            (df_comb.loc[end, 'Tenor']-df_comb.loc[start, 'Tenor'])\
            *(df_comb.loc[mid, 'Tenor']-df_comb.loc[start, 'Tenor'])
    return Df_mid

def PV_Fix(id):
    return np.sum(df_comb.loc[:id, 'OIS_DF'])*df_comb.loc[id, 'IRS']
  
def PV_Float(id, N, x):
    temp = df_comb.loc[id-N+1, 'OIS_DF']*Libor_DF(df_comb.loc[id-N, 'LIBOR_DF'],Interp(id-N,id,id-N+1,x))
    for i in range(N-1):
        temp += df_comb.loc[id-(N-i-2), 'OIS_DF']*Libor_DF(Interp(id-N,id,id-(N-i-1),x),Interp(id-N,id,id-(N-i-2),x))
    return temp

def Forward_LIBOR(id, N):
    for i in range(N):
        df_comb.loc[id-(N-i-1),'F_LIBOR'] = Libor_DF(df_comb.loc[id-(N-i),'LIBOR_DF'],df_comb.loc[id-(N-i-1), 'LIBOR_DF'])

def Libor_DF(D1,D2,delta = 0.5):
    L1 = (D1/D2-1)/delta
    return L1

# FIND FIRST DISCOUNT FACTOR
PV_Flt = df_comb.loc[0, 'OIS_DF']*df_comb.loc[0, 'F_LIBOR']
df_comb.loc[1,'LIBOR_DF'] = brentq(lambda x: \
           PV_Fix(1)- (PV_Flt+df_comb.loc[1,'OIS_DF']*Libor_DF(df_comb.loc[0,'LIBOR_DF'],x)),1e-6, 1.0)
df_comb.loc[1,'F_LIBOR'] = Libor_DF(df_comb.loc[0,'LIBOR_DF'],df_comb.loc[1,'LIBOR_DF'])               
PV_Flt += df_comb.loc[1,'OIS_DF']*df_comb.loc[1,'F_LIBOR']
###############################################################################
for id,iid in zip(index1,index2):
    N = id-iid
    df_comb.loc[id, 'LIBOR_DF'] = brentq(lambda x: PV_Fix(id)- (PV_Flt+PV_Float(id,N,x)),1e-6,1)
    df_comb.loc[iid:id, 'LIBOR_DF'] = df_comb.loc[iid:id, 'LIBOR_DF'].interpolate(method='linear')
    Forward_LIBOR(id, N)
    PV_Flt += np.sum(df_comb.loc[id-N+1:id, 'OIS_DF']*df_comb.loc[id-N+1:id, 'F_LIBOR'])

plt.figure(figsize=(8,6))
plt.plot(df_comb['Tenor'],df_comb['OIS_DF'],'b',linewidth=2,label='OIS Discount')
plt.plot(df_comb.loc[: , 'Tenor'],df_comb.loc[: ,'LIBOR_DF'],'r--', linewidth=2)
plt.title('Bootstrapping', color = 'k')
plt.xlabel('Tenor')
plt.ylabel('Discount')
plt.legend()
#%%
def PVBP(expiry,tenor,delta):
    m = int(tenor/delta)
    summa = 0
    for i in range(1,m+1):
        summa += OISDifInterp(expiry + i*delta)
    return summa*delta

def LiborDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['LIBOR_DF'])
    return np.interp(tenor, xp, yp)

def OISDifInterp(tenor):
    xp = [0] + list(df_comb['Tenor'])
    yp = [1] + list(df_comb['OIS_DF'])
    return np.interp(tenor, xp, yp)

def floatLeg(expiry,tenor,delta):
    m = int(tenor/delta)
    summa = 0
    for i in range(1,m+1):
        loc = expiry + i*delta
        summa += OISDifInterp(loc)*Libor_DF(LiborDifInterp(loc-delta),LiborDifInterp(loc),delta)
    return summa*delta
    
def Swaption(expiry, tenor, delta):
    return floatLeg(expiry,tenor,delta)/PVBP(expiry,tenor,delta)

df_ForwardSwap = pd.read_excel('IR Data.xlsx', 'Swaption', skiprows=[0,1], parse_cols=[0,1] )
df_ForwardSwap = df_ForwardSwap.assign(Swap_Rate=np.nan)

for i in df_ForwardSwap.index:
    expiry = int(df_ForwardSwap.loc[i, 'Expiry'][:-1])
    tenor  = int(df_ForwardSwap.loc[i, 'Tenor'][:-1])
    df_ForwardSwap.loc[i, 'Swap_Rate'] = Swaption(expiry, tenor, 0.5)
    
df_swaption = pd.DataFrame(data = np.array(df_ForwardSwap['Swap_Rate']).reshape(3,5),
                        index   = [1,5,10], 
                        columns = [1,2,3,5,10])
print(df_swaption)
# export for part II, III
df_ForwardSwap.to_csv('df_ForwardSwap.csv', sep=',')
df_comb.to_csv('df_comb.csv', sep=',')
df_swaption.to_csv('df_swaption.csv', sep=',')