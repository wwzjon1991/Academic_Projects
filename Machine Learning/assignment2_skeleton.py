##########
# Part 1 #
##########

import numpy as np
import pandas as pd
from sklearn import datasets, metrics, feature_extraction, naive_bayes, svm, model_selection, ensemble, tree
import matplotlib.pyplot as plt
import time



data_train = datasets.fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes'))
data_test = datasets.fetch_20newsgroups(subset = 'test', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes'))
categories = data_train.target_names
target_map = {}
for i in range(len(categories)):
    if 'comp.' in categories[i]:
        target_map[i] = 0
    elif 'rec.' in categories[i]:
        target_map[i] = 1
    elif 'sci.' in categories[i]:
        target_map[i] = 2
    elif 'misc.forsale' in categories[i]:
        target_map[i] = 3
    elif 'talk.politics' in categories[i]:
        target_map[i] = 4
    else:
        target_map[i] = 5

tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x_train = tfidf_vectorizer.fit_transform(data_train.data)
x_test = tfidf_vectorizer.transform(data_test.data)
y_train = [target_map[i] for i in data_train.target]
y_test = [target_map[i] for i in data_test.target]

#%%
##########
# Part 1 #
##########

from sklearn import linear_model


def my_softmax (x, coef, intercept):
    upper = np.exp(x*coef.T) *np.exp( intercept)
    lower = np.sum(upper, axis=1)
    return upper/lower[:,None]
    
logit = linear_model.LogisticRegression(multi_class = 'multinomial',\
                                        solver = 'newton-cg', C = 0.01)
logit.fit(x_train, y_train)

my_predict_proba = my_softmax(x_test, logit.coef_, logit.intercept_)
print(np.linalg.norm(my_predict_proba - logit.predict_proba(x_test)))

#%%
df_Qns1 = pd.DataFrame(index = ['Softmax', 'Precision', 'Recall', 'F1 score'], \
             columns = ['C=0.01','C=0.1','C=1','C=10','C=100'], dtype = 'float')

for col, cx in enumerate([0.01,0.1,1,10,100]):
    logit = linear_model.LogisticRegression(multi_class = 'multinomial',\
                                            solver = 'newton-cg', C = cx)
    
    logit.fit(x_train, y_train) 
    y_pred = logit.predict(x_test)

    my_predict_proba = my_softmax(x_test, logit.coef_, logit.intercept_)    
    
    df_Qns1.iloc[0, col] = np.linalg.norm(my_predict_proba- logit.predict_proba(x_test))
    df_Qns1.iloc[1, col] = metrics.precision_score(y_test, y_pred, average='weighted')
    df_Qns1.iloc[2, col] = metrics.recall_score(y_test, y_pred, average='weighted')
    df_Qns1.iloc[3, col] = metrics.f1_score(y_test, y_pred, average='weighted')

#%%
##########
# Part 2 #
##########
cat = ['Category 1','Category 2','Category 3','Category 4','Category 5', 'Category 6']
logit = linear_model.LogisticRegression(multi_class = 'multinomial',\
                                        solver = 'newton-cg', C = 1)
logit.fit(x_train, y_train) 

#print(logit.coef_)

df_Qns2 = pd.DataFrame(index =cat ,columns = range(10))
def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        df_Qns2.loc[category, :] = feature_names[top10]
        #print("%s: %s" % (category, " ".join(feature_names[top10])))
    
    return    

show_top10(logit, tfidf_vectorizer, cat)
df_Qns2 = df_Qns2.transpose()

#%%
##########
# Part 3 #
##########

from sklearn import decomposition

tSVD = decomposition.TruncatedSVD(n_components = 100, n_iter = 20, random_state = 2018)
xr_train = tSVD.fit_transform(x_train)
xr_test = tSVD.transform(x_test)

print(xr_train.shape , xr_test.shape)

#%%
##########
# Part 4 #
##########

parameters = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['poly'], 'degree': [2, 3], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['rbf'], 'gamma': ['auto', 0.5, 1, 2], 'C': [0.1, 1, 10, 100]}\
    ]

clfx = model_selection.GridSearchCV(svm.SVC(), parameters, \
                        scoring = metrics.make_scorer(metrics.f1_score, average = 'weighted'),
                        cv = None)
clfx.fit(x_train, y_train)


print('best index:', clfx.best_index_)
print('best score:', clfx.best_score_)
print('best parameters: ', clfx.best_params_)
df_clfx = pd.DataFrame(clfx.cv_results_)
print('time taken:', df_clfx.loc[clfx.best_index_, 'mean_fit_time'])

df_Qns4 = pd.DataFrame(index = ['Original Data', 'Redcued Data'],\
                       columns = ['Time taken','Best F1 score',\
                                  'Kernel','Kernel parameters','C'])

df_Qns4.loc['Original Data', 'Time taken'] = df_clfx.loc[clfx.best_index_, 'mean_fit_time']
df_Qns4.loc['Original Data', 'Best F1 score'] = clfx.best_score_
df_Qns4.loc['Original Data', 'Kernel'] = list(clfx.best_params_.values())[2]
df_Qns4.loc['Original Data', 'Kernel parameters'] = \
                        list(clfx.best_params_.keys())[1]+": "+str(list(clfx.best_params_.values())[1])
df_Qns4.loc['Original Data', 'C'] = list(clfx.best_params_.values())[0]

#%%
parameters = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['poly'], 'degree': [2, 3], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['rbf'], 'gamma': ['auto', 0.5, 1, 2], 'C': [0.1, 1, 10, 100]}\
    ]

clfxr = model_selection.GridSearchCV(svm.SVC(), parameters, \
                        scoring = metrics.make_scorer(metrics.f1_score, average = 'weighted'),
                        cv = None)
clfxr.fit(xr_train, y_train)
    
print('best index:', clfxr.best_index_)
print('best score:', clfxr.best_score_)
print('best parameters: ', clfxr.best_params_)
df_clfxr = pd.DataFrame(clfxr.cv_results_)
print('time taken:', df_clfxr.loc[clfxr.best_index_, 'mean_fit_time'])


df_Qns4.loc['Redcued Data', 'Time taken'] = df_clfxr.loc[clfxr.best_index_, 'mean_fit_time']
df_Qns4.loc['Redcued Data', 'Best F1 score'] = clfxr.best_score_
df_Qns4.loc['Redcued Data', 'Kernel'] = list(clfxr.best_params_.values())[2]
df_Qns4.loc['Redcued Data', 'Kernel parameters'] = \
                        list(clfxr.best_params_.keys())[1]+": "+str(list(clfxr.best_params_.values())[1])
df_Qns4.loc['Redcued Data', 'C'] = list(clfxr.best_params_.values())[0]
#%%

# Grid Search

def my_linspace (min_value, max_value, steps):
    diff = max_value - min_value
    return np.linspace (min_value - 0.1 * diff, max_value + 0.1 * diff, steps)

data = np.loadtxt('nonlinear.txt', delimiter = ',')
x = data[:,:2]
y = data[:,2].astype(int)

color = ['blue', 'red']
y_color = [color[i] for i in y]

parameters = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['poly'], 'degree': [2, 3], 'C': [0.1, 1, 10, 100]},\
    {'kernel': ['rbf'], 'gamma': ['auto', 0.5, 1, 2], 'C': [0.1, 1, 10, 100]}\
    ]


clf = model_selection.GridSearchCV(svm.SVC(), parameters, \
                        scoring = metrics.make_scorer(metrics.f1_score, average = 'weighted'),
                        cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2018))

clf.fit(x, y)

print('best index:', clf.best_index_)
print('best score:', clf.best_score_)
print('best parameters: ', clf.best_params_)
results = pd.DataFrame(clf.cv_results_)
print('time taken:', results.loc[clf.best_index_, 'mean_fit_time'])

#%%
##########
# Part 5 #
##########
df_Qns5 = pd.DataFrame(index = ['Original Data', 'Reduced Data'], \
                       columns =['Bagging', 'Random Forest', 'AdaBoost', 'Gradient Boosting'], dtype = 'float')
###########
# Bagging #
###########
# Original     max_samples = 0.5, max_features = 1, oob_score = True,
x_bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 20),  n_estimators=50,\
                                       random_state = 2018)
x_bagging.fit(x_train, y_train)
y_bagging_pred = x_bagging.predict(x_test)

print('Bagging:',metrics.f1_score(y_test, y_bagging_pred, average='weighted'))
df_Qns5.loc['Original Data', 'Bagging'] = metrics.f1_score(y_test, y_bagging_pred, average='weighted')


# Reduced
xr_bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 20),  n_estimators=50,\
                                      random_state = 2018)
xr_bagging.fit(xr_train, y_train)
yr_bagging_pred = xr_bagging.predict(xr_test)

print('Bagging:',metrics.f1_score(y_test, yr_bagging_pred, average='weighted'))
df_Qns5.loc['Reduced Data', 'Bagging'] = metrics.f1_score(y_test, yr_bagging_pred, average='weighted')

#################
# Random Forest #
#################
# Original
x_rforest = ensemble.RandomForestClassifier(n_estimators=50, max_depth = 20,\
                                            random_state = 2018)
x_rforest.fit(x_train, y_train)
y_rforest_pred = x_rforest.predict(x_test)

print('Random Forest:',metrics.f1_score(y_test, y_rforest_pred, average='weighted')) 
df_Qns5.loc['Original Data', 'Random Forest'] = metrics.f1_score(y_test, y_rforest_pred, average='weighted')

# Reduce
xr_rforest = ensemble.RandomForestClassifier(n_estimators=50, max_depth = 20,\
                                            random_state = 2018)
xr_rforest.fit(xr_train, y_train)
yr_rforest_pred = xr_rforest.predict(xr_test)

print('Random Forest:',metrics.f1_score(y_test, yr_rforest_pred, average='weighted')) 
df_Qns5.loc['Reduced Data', 'Random Forest'] = metrics.f1_score(y_test, yr_rforest_pred, average='weighted')

############
# AdaBoost #
############
# Original
x_adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20),\
                                       n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)
x_adaboost.fit(x_train, y_train)
y_adaboost_pred = x_adaboost.predict(x_test)
print('AdaBoost:',metrics.f1_score(y_test, y_adaboost_pred, average='weighted')) 
df_Qns5.loc['Original Data', 'AdaBoost'] = metrics.f1_score(y_test, y_adaboost_pred, average='weighted')

# Reduce
xr_adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20),\
                                       n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)
xr_adaboost.fit(xr_train, y_train)
yr_adaboost_pred = xr_adaboost.predict(xr_test)
print('AdaBoost:',metrics.f1_score(y_test, yr_adaboost_pred, average='weighted')) 
df_Qns5.loc['Reduced Data', 'AdaBoost'] = metrics.f1_score(y_test, yr_adaboost_pred, average='weighted')

##################
# Gradient Boost #
##################
# Original
x_gboost = ensemble.GradientBoostingClassifier(n_estimators = 50,max_depth =20, \
                                             random_state = 2018)
x_gboost.fit(x_train, y_train)
y_gboost_pred =  x_gboost.predict(x_test)

print('Gradient Boost:',metrics.f1_score(y_test, y_gboost_pred, average='weighted')) 
df_Qns5.loc['Original Data', 'Gradient Boosting'] = metrics.f1_score(y_test, y_gboost_pred, average='weighted')

# Reduced
xr_gboost = ensemble.GradientBoostingClassifier(n_estimators = 50,max_depth =20, \
                                             random_state = 2018)
xr_gboost.fit(xr_train, y_train)
yr_gboost_pred =  xr_gboost.predict(xr_test)

print('Gradient Boost:',metrics.f1_score(y_test, yr_gboost_pred, average='weighted')) 
df_Qns5.loc['Reduced Data', 'Gradient Boosting'] = metrics.f1_score(y_test, yr_gboost_pred, average='weighted')

print(df_Qns5)





