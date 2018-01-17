# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:39:08 2017

@author: Jon Wee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# all the data from excel

# Data files
industrial_portfolios = 'Exam_Q1.xlsx'
Market_Returns = 'Market_Returns.xlsx'
Risk_Factors = 'Risk_Factors.xlsx'

# columns names
Ind_cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops','Hlth', 'Utils', 'Other']

# Industrial Portfolios
Data_Ind = pd.read_excel(industrial_portfolios)
Data_Ind = Data_Ind.drop('Date', 1)
"""
# Market
Data_Mkt = pd.read_excel(Market_Returns)
Data_Mkt = Data_Mkt.drop('Date', 1)
"""
# Factors
Data_Factors = pd.read_excel(Risk_Factors)
Data_Factors = Data_Factors.drop('Date', 1)

# Risk Free
Data_Rf = pd.DataFrame(Data_Factors['Rf'])

#setting up matrix for covariance matrix = V
V = np.cov(Data_Ind.T)

#setting up matrix for average returns 
average_returns = np.mean(Data_Ind, axis=0)
average_returns = np.resize(average_returns, (len(average_returns),1))

samples = 100000

#generating random standardized portfolio weights
PortfolioWeights = np.random.rand(samples, 10)
SampleSum = np.sum(PortfolioWeights, axis=1)
SampleSum = np.resize(SampleSum, (len(SampleSum),1))
SampleSum = np.repeat(SampleSum, 10, axis = 1)

PortfolioWeights = PortfolioWeights / SampleSum

#computing for each sample portfolio mean return and portfolio standard deviation
mean_portfolio_return = []
portfolio_StDev = []

for count in range(samples):
    mean_portfolio_return = np.append(mean_portfolio_return, np.dot(average_returns.T, PortfolioWeights[count]))
    portfolio_StDev = np.append(portfolio_StDev, np.sqrt(np.dot(np.dot(PortfolioWeights[count],V),PortfolioWeights[count].T)))

t = (mean_portfolio_return-0.13)/portfolio_StDev

#scatter plot for portfolios (monte carlo simulation)
plt.figure(figsize=(12, 8))
plt.subplot(1,1,1)
plt.scatter(portfolio_StDev, mean_portfolio_return, c=t, cmap='viridis_r', edgecolor = 'black', linewidths = 0.7)
plt.xlabel("Standard Deviation") ; plt.ylabel("Portfolio Return") ; plt.title("Monte Carlo Simulation")
axes = plt.gca() ; axes.set_ylim([0.6,1.1]) ; axes.set_xlim([3,6])
ax = plt.colorbar()
ax.set_label('Sharpe Ratio', rotation = 270, labelpad = 15)