# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 09:45:18 2017

@author: christophert

pca_yc
"""

from __future__ import  print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Read in data
data_filename = "DTYCR.xlsx"
data = pd.read_excel(data_filename)

# exclude the date column
cols = np.shape(data)[1]
yc = data.iloc[:, 1:cols]   
yc = np.asarray(yc)

# delete rows with any NaN
n  = len(yc)
i = 0
while i < n:
   if np.isnan(yc[i,:]).any() == True:
      #print (i)
      yc = np.delete(yc, i, 0)
      n -= 1
   else:
      i += 1

# De-mean the time series
m = yc.mean()
dyc = yc - m
dyct = yc.T

# PCA by eigenvalue analysis
cov = np.cov(dyct, ddof=1)
cov = np.asmatrix(cov)
eigenvalue,eigenvector = np.linalg.eig(cov)

# Use scikit-learn
pca = PCA(n_components=11)
pca.fit(dyc)

pca_score = pca.explained_variance_ratio_
v = pca.components_

# Compare with own implementation
v =np.asmatrix(v)
V = v.T
print(V[:,0])
print('\n')
print(eigenvector[:,0])

# Plot the first three PCs 
plt.plot(V[:,0], '-ro', label='level')
plt.plot(V[:,1], '-^k', label = 'slope')
plt.plot(V[:,2], '-xb', label = 'curvature')

tenor = list(data)
x = np.linspace(0, 10, 11)
xticks = tenor[1:12]
plt.xticks(x, xticks)
plt.grid()
plt.legend(loc = 3)
plt.savefig('pca.yc.png', dpi=300)
plt.show()

plt.close()








