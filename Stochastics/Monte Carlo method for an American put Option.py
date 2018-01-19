# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:43:25 2018

@author: Jon Wee
"""
import numpy as np

def myLSMput(K, r, dt, S):
    #M, N
    M,n=S.shape
    N=n-1
    #1
    x=(np.ones(M)*N).astype(int)
    #2
    IV=np.maximum(K-S,0)
    #3
    for j in range(N-1, 1-1, -1):
        #3.1
        idx=np.arange(M)[IV[:,j]>0]
        #3.2
        X=S[idx, j]
        Y=IV[idx,x[idx]]*np.exp(-r*(x[idx]-j)*dt)
        #3.3
        P=np.polyfit(X,Y,2)
        #3.4 (1)
        for i in idx:
            if IV[i,j]>np.polyval(P,S[i,j]):
                x[i]=j
        
    return np.mean(IV[range(M),x]*np.exp(-r*x*dt))