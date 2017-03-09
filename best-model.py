#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:06:01 2017

@author: joost
"""

import openml as oml
# This is a temporary read-only OpenML key. Replace with your own key later. 
oml.config.apikey = '11e82c8d91c5abece86f424369c71590'

from openml import tasks

task = tasks.get_task(145677)

X, y = task.get_X_and_y()

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


clf_pipe = Pipeline(
        [
                ('scaling', StandardScaler()),
#                ('selector', GenericUnivariateSelect(mode='k_best', param=170)),
                 ('classifier', VotingClassifier([
                         ('extra', ExtraTreesClassifier(n_estimators=200, criterion='entropy', bootstrap=True)),
                      ('rf', RandomForestClassifier(
                            n_estimators=100, n_jobs=-1, max_features=170)
                      ),
                 ], voting='soft'))
        ]
)

a = cross_val_score(clf_pipe, X, y, cv=10, scoring='roc_auc', n_jobs=5, verbose=3)
print(a, a.mean())