import sys
import os
import pandas as pd
import numpy as np

import pickle
from sklearn import preprocessing, metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from torch import std

import ml_models.svm_models, ml_models.rf_models, ml_models.logreg_models

def label_class(row, option):
        
    if option == 'PETandDX':
        if row['AV45+'] == 1 and row['DX'] == 'Dementia':
            return 1
        if row['AV45+'] == 0 and row['DX'] == 'CN':
            return 0
        if np.isnan(row['AV45+']):
            if row['PIB+'] == 1 and row['DX'] == 'Dementia':
                return 1
            if row['PIB+'] == 0 and row['DX'] == 'CN':
                return 0
    
    if option == 'PET':
        if row['AV45+'] == 1:
            return 1
        if row['AV45+'] == 0:
            return 0
        if np.isnan(row['AV45+']):
            if row['PIB+'] == 1:
                return 1
            if row['PIB+'] == 0:
                return 0
    
    if option == 'LOAD':
        if row['Phenotype'] == 2:
            return 1
        if row['Phenotype'] == 1:
            return 0

def create_class_ADNI(df, option):

    
    df['y'] = df.apply (lambda row: label_class(row, option), axis=1)

    to_drop = list(df.columns)[-27:-1]
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    # print('Class distribution:')
    # print(df['y'].value_counts())

    return df_notna

def create_class_LOAD(df):

    
    df['y'] = df['Phenotype']
    to_drop = ['FID', 'father_IID', 'mother_IID', 'Sex', 'Phenotype']
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    df_notna['y'].replace({2:1, 1:0}, inplace=True)

    # print('Class distribution:')
    # print(df_notna['y'].value_counts())

    return df_notna

def baseline_model_custom(target, infile):

    result_df = pd.DataFrame(columns = ['target', 'dataset', 'model', 'acc', 'pre', 'rec', 'f1', 'auc', 'cm'])
    data = pd.read_csv(infile, index_col = 0)
    data_wclass = create_class_ADNI(data, target)

    x = data_wclass.drop(columns=['y'])
    x = x['APOE']

    y = data_wclass['y']
    x.index = x.index.str.upper()

    for i in range(10):

        i += 1

        f = open(f'data/splits/10Fold_CV_{target}/k{i}_{target}.pkl', 'rb')
        split = pickle.load(f)
        f.close()

        tr_idx = split['train']
        te_idx = split['valid']

        x_train, x_test = x.iloc[tr_idx], x.iloc[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        logreg = LogisticRegression()
        logreg.fit(x_train_scaled, y_train)

        y_prob = logreg.predict_proba(x_test_scaled)

        y_pred = logreg.predict(x_test_scaled)
        acc = metrics.accuracy_score(y_test, y_pred)
        pre = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        f1  = metrics.f1_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_prob[:, 1])
        cm = metrics.confusion_matrix(y_test, y_pred).tolist()

        result_df.loc[len(result_df)] = [target, 'Only APOE', 'Logistic Regression', acc, pre, rec, f1, auc, cm]

    result_df.to_csv(f'results/baseline_models_{target}.csv')


def baseline_model(x, y):

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    auc_scores = []
    k=1
    for tr_idx, te_idx in cv.split(x):

        print(k)

        x_train, x_test = x.iloc[tr_idx], x.iloc[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        logreg = LogisticRegression()
        logreg.fit(x_train_scaled, y_train)

        y_prob = logreg.predict_proba(x_test_scaled)

        y_pred = logreg.predict(x_test_scaled)
        acc = metrics.accuracy_score(y_test, y_pred)
        pre = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        f1  = metrics.f1_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_prob[:, 1])

        print('Acc.', acc)
        print('Pre.', pre)
        print('Rec.', rec)
        print('F1.', f1)
        print('AUC.', f1)
        print()
        # skplt.metrics.plot_roc_curve(y_test, y_prob)
        # plt.show()

        # cm = metrics.confusion_matrix(y_test, y_pred)
        # print()
        # print('Confusion matrix:\n', cm)

        # print()
        # print(metrics.classification_report(y_test, y_pred))
        k += 1

        auc_scores.append(auc)
    
    auc_scores = np.array(auc_scores)
    avg_auc_score = np.mean(auc_scores)
    std_auc_score = np.std(auc_scores)
    print(avg_auc_score, '+-', std_auc_score)


def main(dataset, network, target, indir, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(f'{outdir}/random_forest'):
        os.makedirs(f'{outdir}/random_forest')

    if not os.path.exists(f'{outdir}/svm'):
        os.makedirs(f'{outdir}/svm')

    if not os.path.exists(f'{outdir}/logreg'):
        os.makedirs(f'{outdir}/logreg')

    # Select dataset to use
    if dataset == 'ADNI': # Use ADNI data
        infile = f'{indir}/AD_PPI_missense_ADNI_labeled.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)
    
    elif dataset == 'ADNI_noAPOE': # Use ADNI data
        infile = f'{indir}/noAPOE/AD_PPI_missense_ADNI_labeled'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)

    elif dataset == 'LOAD':
        infile = f'{indir}/AD_PPI_missense_LOAD_labeled.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)

    # # Run parameters
    # split = 0.1 # Percentage of test set
    # n = 3       # Number of folds for GridSearchCV
    # print()
    # print(f'Classification with {target} - Split {split*100}% - GridSearchCV {n}')

    # Data preprocessing
    x = data_wclass.drop(['y'], axis=1)
    y = data_wclass['y']

    # # Load the split used for GNNs models.
    # x.reset_index(inplace=True)
    # x = x.drop(['index'], axis=1)
    # print(x)

    all = []

    for i in range(10):

        i += 1

        f = open(f'data/splits/10Fold_CV_{target}/k{i}_{target}.pkl', 'rb')
        split_dict = pickle.load(f)
        f.close()

        tr_idx = split_dict['train']
        te_idx = split_dict['valid']

        x_train, x_test = x.iloc[tr_idx], x.iloc[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # Data scaling (only for models which depend on it!)
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Random Forests (RFs) GridSearchCV models
        rf, rf_results = ml_models.rf_models.base_model_RF(x_train, x_test, y_train, y_test, i)
        # rf_best, rf_best_results = ml_models.rf_models.grid_search_RF(x_train, x_test, y_train, y_test, target, n)
        # pickle.dump(rf_best, open(f'{outdir}/random_forest/{target}_{disease}_{network}_rf_best_model.csv', 'wb'))

        # Support Vector Machines (SVMs) GridSearchCV models
        svm_lin, svm_lin_results = ml_models.svm_models.base_model_SVM(x_train_scaled, x_test_scaled, y_train, y_test, 'linear', i)
        svm_rbf, svm_rbf_results = ml_models.svm_models.base_model_SVM(x_train_scaled, x_test_scaled, y_train, y_test, 'rbf', i)
        # svm_lin_best, svm_lin_best_results = ml_models.svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, target, n, 'linear')
        # svm_rbf_best, svm_rbf_best_results = ml_models.svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, target, n, 'rbf')
        # pickle.dump(svm_lin_best, open(f'{outdir}/svm/{target}_{disease}_{network}_lin_best_model.csv', 'wb'))
        # pickle.dump(svm_rbf_best, open(f'{outdir}/svm/{target}_{disease}_{network}_rbf_best_model.csv', 'wb'))

        # Logistic Regression (LogReg)
        logreg, logreg_results = ml_models.logreg_models.base_model_LogReg(x_train_scaled, x_test_scaled, y_train, y_test, i)

        # All models results
        results = [logreg_results, svm_lin_results, svm_rbf_results, rf_results]
        results = pd.DataFrame(results, columns = ['acc', 'pre', 'rec', 'f1', 'auc', 'cm', 'fold', 'model'])
        all.append(results)

    results_all = pd.concat(all, axis=0)
    # print(results_all)
    results_all.to_csv(f'results/results_nonGNN_models_{target}.csv')


if __name__ == '__main__':
    
    # Options
    DS = str(sys.argv[1])
    DIS = str(sys.argv[2]) 
    NET = str(sys.argv[3])
    DIR = str(sys.argv[4])
    TAR = str(sys.argv[5])

    # DIS = 'AD'  # or ND
    # NET = 'PPI' # or Multi
    # DIR = '/home/laura/Documents/DATASETS/table_datasets'

    # Input and output directories
    INDIR = 'data'
    OUTDIR = f'results/results_missense_{TAR}'

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    if not os.path.exists(f'{OUTDIR}/random_forest'):
        os.makedirs(f'{OUTDIR}/random_forest')
    
    if not os.path.exists(f'{OUTDIR}/svm'):
        os.makedirs(f'{OUTDIR}/svm')
    
    if not os.path.exists(f'{OUTDIR}/logreg'):
        os.makedirs(f'{OUTDIR}/logreg')

    pet_results = main('ADNI', 'AD', 'PPI', 'PET')
