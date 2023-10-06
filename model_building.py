# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:54:15 2023

@author: faruk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("data_eda.csv")


# relevant columns

df_relevant = df[["avg_salary","Rating", "Size", "Type of ownership",
                  "Industry","Sector","Revenue","comp_num",
                  "hourly","employer_pro","job_state",
                  "same_state","age","python_des","aws_des",
                  "spark_des","azure_des","hadoop_des","scikit-learn_des",
                  "keras_des","google cloud_des","seniority","job_simp","desc_len"]]

# get dummy data

df_dum = pd.get_dummies(df_relevant)
# train-test split
X=df_dum.iloc[ : , 1:]
y=df["avg_salary"].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# multiple linear regression


import statsmodels.api as sm

X_sm=X=sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


from sklearn.linear_model import LinearRegression , Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score


lm = LinearRegression()

lm.fit(X_train , y_train)

np.mean(cross_val_score(lm, X_train , y_train , scoring="neg_mean_absolute_error",cv=2 ))




lm_l=Lasso(alpha=0.15)
lm_l.fit(X_train , y_train)


# lasso regression 



alpha=[]
error=[]

for i in range(1,100):
    alpha.append(i/100)
    lm_lasso=Lasso(alpha=i/100)
    error.append(np.mean(cross_val_score(lm_lasso, X_train , y_train , scoring="neg_mean_absolute_error",cv=3) ))

plt.plot(alpha , error)
plt.show()

min_index = error.index(max(error))
print(alpha[min_index])
#alpha = 0.15

# random forest

from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()

np.mean(cross_val_score(rf, X_train , y_train , scoring="neg_mean_absolute_error",cv=3))
# cross validation score is -14.82 which is best algorithm for this dataset

#model tunning

from sklearn.model_selection import GridSearchCV

parameters = {"n_estimators":range(10,300,10) ,"criterion":['squared_error'],  "max_features":("auto","sqrt","log2")}

gs = GridSearchCV(rf, parameters , scoring="neg_mean_absolute_error",cv=3 )


gs.fit(X_train, y_train)

gs.best_score_


#test
tpred_lm=lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)


mean_absolute_error(y_test, tpred_lm )
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf )


