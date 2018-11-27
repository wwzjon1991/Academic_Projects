# -*- coding: utf-8 -*-
"""
Created on June 1 2017

@author: Christopher Ting
"""
from __future__ import  print_function, division
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


# Parameters
number = 500
angle = np.pi/6  # 30 degrees
lb, ub = -10, 10 

# Generate a line 
x = np.linspace(lb, ub, number)
slope = np.tan(angle)
y = x* slope

# Create a figure
plt.figure(figsize=(ub,ub))
ax = plt.subplot(1,1,1)

# Draw an error of first principal component
arr = plt.arrow(x[0]+1, y[0]+1*slope, x[-1]+8, y[-1]+8*slope, \
head_width=0.25, head_length=0.5, fc='red', linewidth=2, edgecolor='red')

# Generate y + noise
noise = np.random.normal(0, 1, number)
noise = noise - np.mean(noise)
y = y + noise

# These two numbers were found by prior runs
sx, sy = 4.0280561122244478, 3.9578197649538502

ax.plot(sx, sy, 'mx', markersize = 10.0, markeredgewidth=3)
ax.plot(x, y, 'kx', markersize = 2)

ax.set_xlim([lb, ub])
ax.set_ylim([lb, ub])


angle2 = np.pi/6 + np.pi/2
slope2 = np.tan(angle2)
y2 = x*slope2

arr = plt.arrow(1.15, slope2*1.15, -2, -slope2*2, \
head_width=0.25, head_length=0.5, fc='blue', linewidth=2, edgecolor='blue')

plt.grid()
ax.annotate('First Principal Component', xy=(7, 4.1), xytext=(2, 8), 
            arrowprops=dict(facecolor='green', shrink=0.05, width=3.0, headwidth=7, 
            linestyle='solid'), fontsize=17)
ax.text(9.2, -0.75, r'$x$',  fontsize=25)
ax.text(-0.75, 9.2, r'$y$',  fontsize=25)
plt.arrow(-10, 0, 19.5, 0, head_width=0.25, head_length=0.5, fc='k', linewidth=2, edgecolor='black')
plt.arrow(0, -10, 0, 19.5, head_width=0.25, head_length=0.5, fc='k', linewidth=2, edgecolor='black')

plt.arrow(sx, sy, -4.6, -slope*4.6, \
head_width=0.0, head_length=0.0, fc='blue', linewidth=1.5, edgecolor='red', linestyle='-.')

plt.arrow(sx, sy, 0.7, 0.7*slope2, \
head_width=0.0, head_length=0.0, fc='black', linewidth=1.5, edgecolor='blue', linestyle='-.')

plt.arrow(sx, sy, -4, -3.9, \
head_width=0.0, head_length=0.0, fc='black', linewidth=1.5, edgecolor='black', linestyle='-.')


plt.savefig('pca_demo1.png', format = 'png', dpi=240)
plt.show()

r = np.zeros(number, dtype=float)
a = np.zeros(number, dtype=float)
for i in range(number):
   r[i] = np.sqrt(x[i]**2 + y[i]**2)
   a[i] = np.arctan(y[i]/x[i] )

aa = a - angle
first = r * np.cos(aa)
second = r * np.sin(aa)

first_var = np.var(first, ddof=1)
second_var = np.var(second, ddof=1)

print("%0.2f %0.2f" % (first_var, second_var))






