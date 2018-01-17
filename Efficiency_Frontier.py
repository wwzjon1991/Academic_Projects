# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:15:43 2017

@author: Jon Wee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

industrial_portfolios = 'Industry_Portfolios.xlsx'

# Industrial Portfolios
Ind_cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops','Hlth', 'Utils', 'Other']
Data_Ind = pd.read_excel(industrial_portfolios)
Data_Ind = Data_Ind[Ind_cols]

# Risk free rate
Rf = 0.13

# Mean Returns
Ret_Ind = Data_Ind.mean()
#Ret_Ind = np.mean(Data_Ind, axis=0)

# Covariance
Cov_Ind = Data_Ind.cov()

#   "e" ones matrix for weights
e = np.ones((10,1))


##########################################################################################################
####### Efficient Frontier  ##############################################################################
# Alpha R' * V^-1 * e
#alpha = np.asscalar((np.matrix(Ret_Ind))*(np.linalg.inv(np.matrix(Cov_Ind)))*e)
alpha = np.asscalar(np.dot(Ret_Ind, np.dot(np.linalg.inv(Cov_Ind), e)))

# Zeta  R' V^-1 R
#zeta = np.asscalar((np.matrix(Ret_Ind))*(np.linalg.inv(Cov_Ind)) * (np.matrix(Ret_Ind).transpose()))
zeta = np.asscalar(np.dot(Ret_Ind, np.dot(np.linalg.inv(Cov_Ind), Ret_Ind)))

#delta   e' * V^-1  *e
#delta = np.asscalar(e.transpose() * (np.linalg.inv(Cov_Ind)) * e)
delta = np.asscalar(np.dot(e.transpose(), np.dot(np.linalg.inv(Cov_Ind), e)))

# Expected Return from 0 to 2 
Rp = np.arange(0,2.01,0.01)
Rp = np.array(Rp)

# For Efficient Frontier,
def Efficient_Front(Rp, Alpha, Zeta, Delta ):
    num1 = 1/Delta
    num2 = Delta/(Zeta*Delta-(Alpha**2))
    eff = []
    for i in Rp:
        x = num1+num2*((i-Alpha/Delta)**2)
        eff = np.append(eff, x )
    return np.sqrt(eff)

# std of efficient frontier
Eff_std = Efficient_Front(Rp, alpha, zeta, delta )
# Efficient Frontier plot with Std(X-axis) and Rp(Y-axis)

# Returns and std for minimum variance portfolio
Rmv=alpha/delta
Rmv_std= Eff_std.min()


##########################################################################################################
####### TANGENCY PORTFOLIO ON Efficient Frontier  ########################################################
# Tangency Portfolio Returns, Standard Dev, Risk Premium, Slope, 
# Lambda,weights of portfolios, 
# Returns
Ret_tg = (alpha*Rf-zeta)/(delta*Rf-alpha)
# Risk Premium
Risk_p_tg = -(zeta-2*alpha*Rf+delta*(Rf**2))/(delta*(Rf-Rmv))
# Standard Deviation
std_tg = -((zeta-2*alpha*Rf+delta*(Rf**2))**0.5)/(delta*(Rf-Rmv))
# Slope
slope_tg = std_tg*(zeta*delta-(alpha**2))/(delta*(Ret_tg-Rmv))


# Efficient Frontier with Riskless asset: - y = mx+c
Riskless_std = np.arange(0,5,0.01)
Eff_Riskless = slope_tg*Riskless_std+Rf 

# Lambda
Lambda_tg = 1/(alpha-delta*Rf) 

# weights of tangency portfolios
w_tg = Lambda_tg*(np.linalg.inv(Cov_Ind))*(np.matrix(Ret_Ind).transpose()-Rf*e)
#print(w_tg)
#wtable = pd.DataFrame(data=w_tg, index=Ind_cols, columns=['Weights(%)'])
##########################################################################################################

#############  orthogonal frontier portfolio         #############################################################################################
# index of orthogonal portfolio
P2_ret = np.searchsorted(Rp, 0.13) # formula
P2_std = Eff_std[P2_ret]

######  PLOT EFFicient Frontier   ########################################################################

plt.figure(figsize=(18 , 12))

# Plot Industrial portfolio frontier
plt.plot(Eff_std, Rp, 'r', label = 'Minimum variance frontier', linewidth = 5)

# Plot Efficient Frontier with riskless asset
plt.plot(Riskless_std, Eff_Riskless, 'k', label = 'Efficient Frontier with riskless asset', linewidth = 5 ) 

# Plot minimum-variance portfolio
plt.plot(Rmv_std, Rmv, 'bo', markersize=25.0)
plt.text( 0.2+Rmv_std, Rmv, 'Minimum variance portfolio: (2.692, 1.0)', fontsize = 15)

# Plot tangent portfolio
plt.plot(std_tg, Ret_tg, 'go', markersize=25.0  )
plt.text( 0.2+std_tg, Ret_tg ,'Tangency portfolio: (3.3628, 1.487 ) ', fontsize = 15)

# Plot Riskless asset 
plt.plot(0, Rf, 'ro', markersize=25.0)
plt.text( 0.2+0, Rf, 'Riskless asset: (0.0, 0.130)', fontsize = 15)

# Plot Riskless asset 
plt.plot(P2_std, Rf, 'g^', markersize=25.0)
plt.text( 0.1+P2_std, Rf, 'Orthogonal Portfolios (4.524,0.13)', fontsize = 15)

plt.grid(False)
plt.xlabel('Standard Deviation', fontsize = 20)
plt.ylabel('Mean Return', fontsize = 20)
plt.title('Project HWA', fontsize = 30)
plt.legend(fontsize = 18)

#################################################################################################

