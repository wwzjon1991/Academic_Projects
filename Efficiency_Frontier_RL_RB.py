# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:27:47 2017

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

# Mean Returns
Ret_Ind = Data_Ind.mean()
#Ret_Ind = np.mean(Data_Ind, axis=0)

# Covariance
Cov_Ind = Data_Ind.cov()

#   "e" ones matrix for weights
e = np.ones((10,1))

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
####### TANGENCY PORTFOLIO2 with different lending and borrowing  #########################################
# Tangency Portfolio Returns, Standard Dev, Risk Premium, Slope, Lambda,weights of portfolios, 

# lending rate
RL = 0.13

# Lending Returns
Ret_tg2 = (alpha*RL-zeta)/(delta*RL-alpha)
# Lending Standard Deviation
std_tg2 = -((zeta-2*alpha*RL+delta*(RL**2))**0.5)/(delta*(RL-Rmv))
# Lending Slope
slope_tg2 = std_tg2*(zeta*delta-(alpha**2))/(delta*(Ret_tg2-Rmv))
# Lending Efficient Frontier with lending rate asset: - y = mx+c
lending_std = np.arange(0,5,0.01)
Eff_lending = slope_tg2*lending_std+RL 
# Lending Lambda
Lambda_tg2 = 1/(alpha-delta*RL) 

# weights of tangency portfolios Lending
w_tg2 = Lambda_tg2*(np.linalg.inv(Cov_Ind))*(np.matrix(Ret_Ind).transpose()-RL*e)


################################################################################################################
# borrowing rate
RB = 0.4
# Lending Returns
Ret_tg3 = (alpha*RB-zeta)/(delta*RB-alpha)
# Lending Standard Deviation
std_tg3 = -((zeta-2*alpha*RB+delta*(RB**2))**0.5)/(delta*(RB-Rmv))
# Lending Slope
slope_tg3 = std_tg3*(zeta*delta-(alpha**2))/(delta*(Ret_tg3-Rmv))
# Lending Efficient Frontier with lending rate asset: - y = mx+c
borrowing_std = np.arange(0,5,0.01)
Eff_borrowing = slope_tg3*borrowing_std+RB 
# Lending Lambda
Lambda_tg3 = 1/(alpha-delta*RB) 

# weights of tangency portfolios Borrowing
w_tg3 = Lambda_tg3*(np.linalg.inv(Cov_Ind))*(np.matrix(Ret_Ind).transpose()-RB*e)


######  PLOT EFFicient Frontier   ########################################################################

plt.figure(figsize=(18 , 12))

# Plot Industrial portfolio frontier
plt.plot(Eff_std, Rp, 'r', label = 'Minimum variance frontier', linewidth = 5)


# Plot minimum-variance portfolio
plt.plot(Rmv_std, Rmv, 'bo', markersize=25.0)
plt.text( 0.2+Rmv_std, Rmv, 'Minimum variance portfolio: (2.692, 1.0)', fontsize = 15)


####################### EXTRA  #############################################################################
# Plot lending tangent portfolio
plt.plot(std_tg2, Ret_tg2, 'ko', label='lending portfolio' , markersize=15.0  )
# Plot Efficient Frontier with riskless asset
plt.plot(lending_std, Eff_lending, 'k', label = 'Efficient Frontier with lending' ) 

plt.plot(0, RL, 'ko', markersize=15.0)
plt.text( 0.2+0, RL, 'lending rate: %s' %RL, fontsize = 15)


# Plot borrowing tangent portfolio
plt.plot(std_tg3, Ret_tg3, 'go', label='borrowing portfolio' , markersize=15.0  )
# Plot Efficient Frontier with riskless asset
plt.plot(borrowing_std, Eff_borrowing, 'g', label = 'Efficient Frontier with borrowing' )

plt.plot(0, RB, 'go', markersize=15.0)
plt.text( 0.2+0, RB, 'borrowing rate: %s' %RB, fontsize = 15)
####################### EXTRA  ##############################################################################

plt.grid(False)
plt.xlabel('Standard Deviation', fontsize = 20)
plt.ylabel('Mean Return', fontsize = 20)
plt.title('Project HWA', fontsize = 30)
plt.legend(fontsize = 18)