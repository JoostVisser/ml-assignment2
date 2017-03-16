#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:06:01 2017

@author: joost
"""

import openml as oml
from openml import tasks, runs

task = tasks.get_task(145677)

X, y = task.get_X_and_y()

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer, PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
import itertools
import xmltodict

# 500 gives 0.852, 200 gives 0.849, 0.850
#clf_pipe = Pipeline(
#        [
#                ('scaling', StandardScaler()),
#                ('selector', GenericUnivariateSelect(mode='k_best', param=250)),
#                ('nn', MLPClassifier(
#                            hidden_layer_sizes=(10),
#                            activation='relu', 
#                            solver='sgd',
#                            alpha=0.3,
#                            learning_rate='adaptive',
#                            max_iter=300,
#                            verbose=1
#                            )
#                )
#        ]
#)

# 0.848
#clf_pipe = Pipeline(
#        [
#                ('scaling', StandardScaler()),
#                ('selector', GenericUnivariateSelect(mode='k_best', param=300)),
#                ('nn', MLPClassifier(
#                            hidden_layer_sizes=(80),
#                            activation='relu', 
#                            solver='sgd',
#                            alpha=3,
#                            learning_rate='adaptive',
#                            max_iter=200,
#                            )
#                )
#        ]
#)
#
#xgb_grid = {
#        'vot__weights': [list((i+1, j+1, k+1, l+1)) for i, j, k, l in itertools.product(range(4), range(4), range(4), range(4))]
#}

clf_grid = {
        'vot__weights': [[1, 2, 2, 3], [1, 1, 1, 1]]
}
#
## New record! 0.88572
#clf_vot = VotingClassifier(
#            [
#                ('extra', ExtraTreesClassifier(n_estimators=512, criterion='entropy', bootstrap=True, max_features="sqrt", min_samples_leaf=2)),
#                ('rf', RandomForestClassifier(
#                        n_estimators=512, n_jobs=1, max_features=0.1, criterion='entropy', bootstrap=True, min_samples_leaf=2)
#                ),
#                ('gradboost', GradientBoostingClassifier(n_estimators=512, max_depth=8, learning_rate=0.05 , max_features="sqrt", min_samples_leaf=2)),
#                ('xgboost', xgb.XGBClassifier(
#                        n_estimators=512, 
#                        max_depth=8, 
#                        silent=True, 
#                        objective="binary:logistic",
#                        learning_rate=0.05,
#                        min_child_weight=2,
#                        nthread=1,
#                        gamma=0,
#                        subsample=0.8,
#                        colsample_bytree=0.9,
#                        reg_lambda=1,
#                        reg_alpha=0)
#                )
#
#            ], voting='soft')
#
##New record! 0.88396s
#clf_pipe = Pipeline ( [
#    ('stsc', StandardScaler()),
#    ('vot', clf_vot)
#    ]
#)
#
#grid = GridSearchCV(clf_pipe, param_distributions=clf_grid, cv=10, n_jobs=8, verbose=2, scoring='roc_auc')
#grid.fit(X, y)
#
#print()
#print(grid.cv_results_)
#
#print("Mean test score: " + str(grid.cv_results_["mean_test_score"]))
#print("Best parameters: {}".format(grid.best_params_))
#print("Best cross-validation score (ROC_AUC): {:.2f}".format(grid.best_score_))

## RESULTS: 1, 2, 2, 3


#
# Results parameter tuning:
# max_depth = 8
# min_child_weight = 2
# Gamma = 0.0
# colsample_bytree = 0.85
# subsample = 0.8 
# reg_alpha=0
# reg_lambda = 0.5
# learning_rate=0.01 || learning_rate=0.05
# Higher n_estimators? --> Lower learning rates

# 0.8796 with learning_rate 0.01, or 0.05
#clf_xgb = xgb.XGBClassifier(
#                        n_estimators=512, 
#                        max_depth=8, 
#                        silent=True, 
#                        objective="binary:logistic",
#                        learning_rate=0.01,
#                        min_child_weight=2,
#                        nthread=1,
#                        gamma=0,
#                        subsample=0.8,
#                        colsample_bytree=0.9,
#                        reg_lambda=1,
#                        reg_alpha=0)

#xgb_grid = {
#        'xgboost__learning_rate': [0.01, 0.05, 0.1, 0.15],
#}
#
#grid = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=5, n_jobs=8, verbose=2, scoring='roc_auc')
#grid.fit(X, y)
#
#print()
#print(grid.cv_results_)
#
#print("Mean test score: " + str(grid.cv_results_["mean_test_score"]))
#print("Best parameters: {}".format(grid.best_params_))
#print("Best cross-validation score (ROC_AUC): {:.2f}".format(grid.best_score_))


# Best classifier:
#clf_vot = VotingClassifier(
#            [
#                ('extra', ExtraTreesClassifier(n_estimators=2048, criterion='entropy', bootstrap=True, max_features="sqrt", min_samples_leaf=2)),
#                ('rf', RandomForestClassifier(
#                        n_estimators=2048, n_jobs=1, max_features=0.1, criterion='entropy', bootstrap=True, min_samples_leaf=2)
#                ),
#                ('gradboost', GradientBoostingClassifier(n_estimators=2048, max_depth=8, learning_rate=0.01 , max_features="sqrt", min_samples_leaf=2)),
#                ('xgboost', xgb.XGBClassifier(
#                        n_estimators=2048, 
#                        max_depth=8, 
#                        silent=True, 
#                        objective="binary:logistic",
#                        learning_rate=0.01,
#                        min_child_weight=2,
#                        nthread=1,
#                        gamma=0,
#                        subsample=0.8,
#                        colsample_bytree=0.9,
#                        reg_lambda=1,
#                        reg_alpha=0))
#
#            ], voting='soft')
#
##New record! 0.88396
#clf_pipe = make_pipeline (
#    StandardScaler(),
#    clf_vot
#)

etc = ExtraTreesClassifier (
    n_estimators=1024, 
    criterion='entropy', 
    bootstrap=True, 
    max_features="sqrt", 
    min_samples_leaf=2
)

rfc = RandomForestClassifier (
    n_estimators=1024, 
    n_jobs=-1,
    max_features=0.1, 
    criterion='entropy', 
    bootstrap=True, 
    min_samples_leaf=2
)

gbc = GradientBoostingClassifier (
    n_estimators=1024, max_depth=8, learning_rate=0.01 , max_features="sqrt", min_samples_leaf=2
)

__version__ = 1

class SecondGradientBoostingClassifier (GradientBoostingClassifier):
    def hello():
        print("Mine turtle")
        

gbc2 = SecondGradientBoostingClassifier (
    n_estimators=1024, max_depth=8, learning_rate=0.02 , max_features="sqrt", min_samples_leaf=2
)

xgb = xgb.XGBClassifier (
        n_estimators=1024, 
        max_depth=8, 
        silent=True, 
        objective="binary:logistic",
        learning_rate=0.02,
        min_child_weight=2,
        nthread=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1,
        reg_alpha=0
)

# New record! 0.8866
clf_vot = VotingClassifier(
            [
               # ('estimators', etc),
                ('voting', rfc),
                ('weights', gbc),
                ('n_jobs', gbc2),
#                ('n_jobs',  xgb)

            ], voting='soft', n_jobs=-1, weights=[2,3,3])

#New record! 0.8866
clf_pipe = make_pipeline (
    StandardScaler(),
    clf_vot
)

a = cross_val_score(clf_pipe, X, y, cv=task.iterate_all_splits(), scoring='roc_auc', n_jobs=5, verbose=3)
print(a, a.mean())


#run = runs.run_task(task, clf_pipe)
#run.publish()
#print("Uploaded run with id %s. Check it at www.openml.org/r/%s" %(run.run_id,run.run_id))
