#!usr/bin/python3

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def compute_metrics(true, prob, prediction):

    acc = metrics.accuracy_score(true, prediction)
    pre = metrics.precision_score(true, prediction)
    rec = metrics.recall_score(true, prediction)
    f1 = metrics.f1_score(true, prediction)
    auc = metrics.roc_auc_score(true, prob[:, 1])
    cm = metrics.confusion_matrix(true, prediction)

    results = [acc, pre, rec, f1, auc, cm]
    return results

def base_model_LogReg(x_train_scaled, x_test_scaled, y_train, y_test):

    print()
    print(f'Base model - LogReg')

    logreg_clf = LogisticRegression()
    logreg_clf.fit(x_train_scaled, y_train)
    y_prob = logreg_clf.predict_proba(x_test_scaled)
    y_pred = logreg_clf.predict(x_test_scaled)

    results = compute_metrics(y_test, y_prob, y_pred)
    results.append('BASE')
    results.append(f'LogReg')

    return logreg_clf, results