# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:47:33 2021

@author: aleks
"""
import pandas as pd
df = pd.read_excel('RAPM.xlsx', engine='openpyxl')

y = df["y"]
y = y.fillna(0)
df.drop(df.columns[[0,1,2,3]], axis = 1, inplace = True)
df = df.fillna(0)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
y = y.values
X = df.values
alpha = [3000]
params = {"alpha": alpha}
ridge = Ridge()
grid = GridSearchCV(ridge, params)
gs = grid.fit(X, y)
clf = grid.best_estimator_
clf = gs.best_estimator_
clf.fit(X, y)

rank = {"ranking": clf.coef_.T}
ranking = pd.DataFrame(data = rank, index = df.columns).sort_values("ranking", ascending = False)

