# -*- coding: utf-8 -*-
"""
Created on June 1 2017

@author: Christopher Ting
"""
from __future__ import  print_function, division
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Collect data
year = range(2007,2017)
x_original = [2.5,0.4,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1]    # Annual return of Stock X in %
y_original = [2.4,0.6,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]    # Annual return of Stock X in %

# Step 2: Demean
mx = np.mean(x_original)
my = np.mean(y_original)

x = np.asarray(x_original - mx)
y = np.asarray(y_original - my)

X = np.matrix([x, y])

# Sep 3: Compute the variance-covariance matrix
Sigma = np.cov(X, ddof=1)

# Step 4: Compute the eigenvalues and eigenvectors 
eigenvalues, eigenvectors = np.linalg.eig(Sigma)

# Step 5: Choose components to form features
features2 = eigenvectors

# Step 6: Get the data back after transformation
fdata2 = features2.dot(X)
fx = np.asarray(fdata2[0])
fy = np.asarray(fdata2[1])
fx = fx[0]
fy = fy[0]

# Visual presentation
plt.plot(x, y, 'bo', markersize=7, label="demeaned")
plt.plot(fx, -fy, 'r^', markersize=7, label="pca transformed")

slope = Sigma[0,1]/np.var(y, ddof=1)
lx = np.linspace(-2, 2, 100)
ly = lx * slope
plt.plot(lx, ly, '-b', linewidth=2)

ly2 = np.zeros(100, dtype=float)
plt.plot(lx, ly2, '-r', linewidth=2)

plt.plot(ly2, lx, '-k', linewidth=1.5)
plt.grid()
plt.legend(loc="upper left")
plt.savefig('pca_demo2.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()



