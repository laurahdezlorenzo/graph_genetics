#!usr/bin/python3

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import svm, metrics

def compute_metrics(true, prob, prediction):

    acc = metrics.accuracy_score(true, prediction)
    pre = metrics.precision_score(true, prediction)
    rec = metrics.recall_score(true, prediction)
    f1 = metrics.f1_score(true, prediction)
    auc = metrics.roc_auc_score(true, prob[:, 1])
    cm = metrics.confusion_matrix(true, prediction)

    results = [acc, pre, rec, f1, auc, cm]
    return results

def base_model_SVM(x_train_scaled, x_test_scaled, y_train, y_test, k):

    print()
    print(f'Base model - SVM {k}')

    svm_clf = svm.SVC(kernel=k, probability=True)
    svm_clf.fit(x_train_scaled, y_train)
    y_prob = svm_clf.predict_proba(x_test_scaled)
    y_pred = svm_clf.predict(x_test_scaled)

    results = compute_metrics(y_test, y_prob, y_pred)
    results.append('BASE')
    results.append(f'SVM {k}')

    return svm_clf, results

def grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, target, n, k):

    print()
    print(f'Grid Search CV - SVM {k}')

    # Grid Search CV of SMV
    if k == 'linear':
        parameters_grid = [{'kernel': [k],
                            'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    elif k == 'rbf':
        parameters_grid = [{'kernel': ['rbf'],
                        'C': [0.01, 0.1, 1, 10, 100, 1000],
                        'gamma': [0.001, 0.01, 0.1, 1]}]
    
    print(parameters_grid)

    # Set model
    skf = StratifiedKFold(n_splits=n,
                            shuffle=True)

    grid_svm_clf = GridSearchCV(svm.SVC(probability=True),
                                parameters_grid,
                                cv=skf,
                                scoring='roc_auc')

    # Train model
    grid_svm_clf.fit(x_train_scaled,
                        y_train)

    print('The best parameters are %s with a score of %0.2f' %
            (grid_svm_clf.best_params_, grid_svm_clf.best_score_))

    # Save best model
    best_svm_clf = grid_svm_clf.best_estimator_

    # Save Grid Search results
    grid_search_results = pd.DataFrame(grid_svm_clf.cv_results_)

    # Make predictions
    y_prob = best_svm_clf.predict_proba(x_test_scaled)
    y_pred = best_svm_clf.predict(x_test_scaled)

    results = compute_metrics(y_test, y_prob, y_pred)
    results.append('BEST')
    results.append(f'SVM {k}')

    return best_svm_clf, results