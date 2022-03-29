#!usr/bin/python3

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics, ensemble

def compute_metrics(true, prob, prediction):

    acc = metrics.accuracy_score(true, prediction)
    pre = metrics.precision_score(true, prediction)
    rec = metrics.recall_score(true, prediction)
    f1 = metrics.f1_score(true, prediction)
    auc = metrics.roc_auc_score(true, prob[:, 1])
    cm = metrics.confusion_matrix(true, prediction)

    results = [acc, pre, rec, f1, auc, cm]
    return results

def base_model_RF(x_train, x_test, y_train, y_test, n):

    rf_clf = ensemble.RandomForestClassifier()
    rf_clf.fit(x_train, y_train)
    y_pred = rf_clf.predict(x_test)
    y_prob = rf_clf.predict_proba(x_test)

    results = compute_metrics(y_test, y_prob, y_pred)
    results.append(n)
    results.append('Random Forest')

    return rf_clf, results

def grid_search_RF(x_train, x_test, y_train, y_test, target, n):

    # Grid Search CV of SMV
    parameters_grid = {
                        'max_features': ['sqrt'],
                        'bootstrap': [True],
                        'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700,
                                            800, 900, 1000]}

    # Set model
    skf = StratifiedKFold(n_splits=n,
                            shuffle=True)

    grid_rf_clf = GridSearchCV(ensemble.RandomForestClassifier(),
                                parameters_grid,
                                cv=skf,
                                scoring='roc_auc')

    # Train model
    grid_rf_clf.fit(x_train, y_train)

    print('The best parameters are %s with a score of %0.2f' %
            (grid_rf_clf.best_params_, grid_rf_clf.best_score_))

    # Save best model
    best_rf_clf = grid_rf_clf.best_estimator_

    # Save Grid Search results
    grid_search_results = pd.DataFrame(grid_rf_clf.cv_results_)

    # Make predictions
    y_prob = best_rf_clf.predict_proba(x_test)
    y_pred = best_rf_clf.predict(x_test)

    results = compute_metrics(y_test, y_prob, y_pred)
    results.append('BEST')
    results.append('Random Forest')

    return best_rf_clf, results