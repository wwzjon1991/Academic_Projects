# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:42:15 2018

@author: Jon Wee
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:02:46 2018

@author: Jon Wee
"""
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from math import sin,e
import pandas as pd
from numpy import array, zeros, diag, diagflat, dot, tril

def func(x):
    x=x/12
    return 300/x*((1+x)**240-1)-400000

solve_x = opt.bisect(lambda x: func(x), a= 1e-5, b=0.2)
#%% Bisection
def Bisection(a,b,tol):
    if func(a)*func(b)>0:
        return "error: f(a)f(b)<0 not satisfied"
           
    else:
        fa=func(a)
        fb=func(b)
        step=0
        while (b-a)/2>tol:
            c=(a+b)/2
            fc=func(c)
            step+=1
            if fc==0:
                return c,step
            elif fc*fa<0:
                b=c
                #fb=fc
            else:
                a=c
                fa=fc
        return c, step
bisection= Bisection(0.01, 1.0, 1e-5)                
print("Bisection Method: ", bisection[0])
#%% Secant
def Secant(x0,x1,tol,kmax):
    fx0 = func(x0)
    fx1 = func(x1)
    steps=2
    for i in range(kmax):
        x2 = x1 - fx1*(x1-x0)/(fx1-fx0)
        fx2 = func(x2)
        if abs(fx2)<tol:

            break
        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2
        steps+=1
        if steps == kmax:
            print("Secant method has not converged")
        
    return x2, steps

secant= Secant(0.01,1.0,1e-5, 1000)                
print("Secant Method: ", secant[0])
#%% False position
def func1(x):
    return 8*sin(x)*(e**-x)-1
#    return x**2-2
def FalsePosition(a,b,tol,kmax):
    steps = 0
    
    for i in range(kmax):
        c = (b*func1(a)-a*func1(b))/(func1(a)-func1(b))
        steps +=1
        if func1(c)==0:
            return c, steps
        elif func1(a)*func1(c)<0:
            b=c
        else:
            a=c
            
 
    
abc,steps = FalsePosition( 0.01, 1, 1e-5, 100) # guess cannot be zero
print(abc,steps)   
#%% Newton root solver
def Newton(x0, tol, kmax):
    def funcprime(x):
        h = 1e-5
        rise = func(x+h)-func(x)
        slope = rise / h
        return slope 
    x1 = x0-(func(x0)/funcprime(x0))
    steps = 1
    
    while abs(x1-x0)>tol:
        x0 = x1
        x1 = x0-(func(x0)/funcprime(x0))
        steps += 1
                
        if steps == kmax:
            print("Newton Method failed to converged")
            return x0,x1,steps
    return x1,steps 

newton = Newton(0.01, 1e-5, 1000)
print("Newton Method: ", newton[0])

#%% Jacobi and Guass Seidel

n = 50
A = np.zeros((n,n))

for i in range(len(A)):
    A[i][i] = 3 #A[row][col]
for i in range(len(A)-1):    
    A[i+1][i] = -1
    A[i][i+1] = -1

B = np.ones((n,))    
B[0] = 2
B[-1]= 2

guess = np.zeros((50,), dtype=float)

def jacobi(A,b,tol,kmax,x):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    n = len(b)

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diagflat(diag(A))
    R = A - D # L+U 
    steps = 0
    # Iterate for kmax times                                                                                                                                                                          
    for i in range(kmax):
        
        x = dot(np.linalg.inv(D) ,b - dot(R,x)) # D^-1*(b-(L+U)*x)
        error = b-np.dot(A,x)
        k=i
        steps +=1
        if np.linalg.norm(error)<tol:
            print("jacobi method has converged")
            break
        elif k==kmax:
            print("jacobi method has not converged")
        
    return x, steps

jac_sol, steps = jacobi(A,B, 1e-5,kmax=50,x=guess)
print("Jacobi Method:", steps,"steps")
print(jac_sol)

def gauss_seidel(A,b,tol,kmax=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diagflat(diag(A))
    L = tril(A,-1)
    U = A-L-D
    
    steps = 0
    # Iterate for kmax times                                                                                                                                                                          
    for i in range(kmax):
        
        x =  dot(np.linalg.inv(D+L) , b-dot(U,x))# D^-1*(b-(L+U)*x)
        error = b-np.dot(A,x)
        k=i
        steps +=1
        if np.linalg.norm(error)<tol:
            print("gauss seidel method has converged")
            break
        elif k==kmax:
            print("gauss seidel method has not converged")
        

    return x, steps

guass_sol, G_steps = gauss_seidel(A,B,1e-5,kmax=50,x=guess)
print("Guass-Seidel: ", G_steps,"steps")
print(guass_sol)

#%% Cubic Spline

x1 = [2,3,4,5,7,10,15]
y1= [6.01,6.11,6.16,6.22,6.32,6.43,6.56]

#x1 = [0,1,2,3,4,5]
#y1 = [3,1,4,1,2,0]

def cubic_spline(x,y):
    n = len(x)
    delta = []
    tri = []
    a = y

    # get delta and tri values
    for i in range(n-1):
        
        delta = np.append(delta, x[i+1]-x[i])
        tri   = np.append(tri, y[i+1]-y[i])
        
    # create matrix
    delta_matrix = np.zeros((n,n))    
    tri_matrix = np.zeros(n,)
    delta_matrix[0,0],delta_matrix[-1,-1] = 1,1
    tri_matrix[0],tri_matrix[-1] = 0,0
        
    for i in range(1,n-1):
        delta_matrix[i,i] = 2*delta[i-1]+2*delta[i]
        delta_matrix[i,i-1] = delta[i-1]
        delta_matrix[i,i+1] = delta[i]
            
        tri_matrix[i] = 3*(tri[i]/delta[i]-tri[i-1]/delta[i-1]) 
            
    c = np.dot(np.linalg.inv(delta_matrix),tri_matrix)
            
    b , d = [],[]
            
    for i in range(n-1):
        d = np.append(d, (c[i+1]-c[i])/(3*delta[i]))
        b = np.append(b, tri[i]/delta[i]-(delta[i]/3*(2*c[i]+c[i+1])))
        
    print("a: ",a)
    print("b: ",b)
    print("c: ",c)
    print("d: ",d)
                
                
    # prepare data for plotting the splines
    xs = np.linspace(min(x),max(x),num=100)
    ys = []
    #for id,i in enumerate(xs):
    for id in range(n-1):
        for i in xs:
            if x[id]<=i<=x[id+1]:
                s = y[id]+b[id]*(i-x[id])+c[id]*(i-x[id])**2+d[id]*(i-x[id])**3
                ys = np.append(ys, s)
    return xs,ys

xs,ys = cubic_spline(x1,y1)
    
plt.figure(figsize=(8 , 6))
plt.plot(x1,y1, 'ro', label='Data Points')
plt.plot(xs,ys,'b', label="cubic spline interpolation")    
plt.title('Cubic Spline Interpolation:Yield Curve', color = 'k')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.legend()
plt.show()    

#%% DR composition
Data = pd.read_excel('Qns5.xlsx')
n = len(Data)


A = np.ones((30,3))
B = np.zeros((30,1))
x = 0
for i in range(0,n,3):
    B[x,0] = Data.loc[i]['Data']
    A[x,1] = Data.loc[i+1]['Data']
    A[x,2] = Data.loc[i+2]['Data']
    x+=1
  
Q,R = np.linalg.qr(A)
# Let transpose(Q)*b=d
d = np.dot(np.transpose(Q),B)

X = np.dot(np.linalg.inv(R),d)

Beta = X
Peter = np.array([[1,12,14]])
Y = np.dot(Peter,Beta)

r = B-np.dot(A,X)
sq_error = np.square(np.linalg.norm(r))
RMSE = np.sqrt(sq_error/len(r))

print("Peter Final test mark is", round(float(Y),2))
print("Peter Total mark is", round(float(Y+12+14),2))
