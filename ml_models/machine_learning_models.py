import sys
import os
import pandas as pd
import numpy as np

import pickle
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import matplotlib.pyplot as plt

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
        if row['LOAD'] == 1:
            return 1
        if row['LOAD'] == 0:
            return 0

def create_class_ADNI(df, option):

    
    df['y'] = df.apply (lambda row: label_class(row, option), axis=1)

    to_drop = list(df.columns)[-27:-1]
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    print('Class distribution:')
    print(df['y'].value_counts())

    return df_notna

def create_class_LOAD(df):

    
    df['y'] = df['Phenotype']
    to_drop = ['FID', 'father_IID', 'mother_IID', 'Sex', 'Phenotype']
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    df_notna['y'].replace({2:1, 1:0}, inplace=True)

    print('Class distribution:')
    print(df_notna['y'].value_counts())

    return df_notna

def baseline_model(split_dict, x, y):

    tr_idx = split_dict['train']
    te_idx = split_dict['test']

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

    # skplt.metrics.plot_roc_curve(y_test, y_prob)
    # plt.show()

    cm = metrics.confusion_matrix(y_test, y_pred)
    print()
    print('Confusion matrix:\n', cm)

    print()
    print(metrics.classification_report(y_test, y_pred))

    return auc

def main(dataset, disease, network, target, indir, outdir):

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
        infile = f'{indir}/{disease}_{network}_missense_with_biomarkers.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)
    
    elif dataset == 'ADNI_noAPOE': # Use ADNI data
        infile = f'{indir}/noAPOE/{disease}_{network}_missense_with_biomarkers.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)

    elif dataset == 'LOAD':
        infile = f'{indir}/{disease}_{network}_missense_with_biomarkers_LOAD.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, target)

    # Run parameters
    split = 0.1 # Percentage of test set
    n = 5       # Number of folds for GridSearchCV
    print()
    print(f'Classification with {target} - Split {split*100}% - GridSearchCV {n}')

    # Data preprocessing
    x = data_wclass.drop(['y'], axis=1)
    y = data_wclass['y']

    # Load the split used for GNNs models.
    x.reset_index(inplace=True)
    x = x.drop(['index'], axis=1)
    print(x)

    f = open(f'data/splits/split_{target}.pkl', 'rb')
    split_dict = pickle.load(f)
    f.close()

    tr_idx = split_dict['train']
    te_idx = split_dict['test']

    x_train, x_test = x.iloc[tr_idx], x.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # Class distribution within each split
    print()
    print(f'Train set\n{y_train.value_counts()}')
    print(f'Test set\n{y_test.value_counts()}')

    # Data scaling (only for models which depend on it!)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Random Forests (RFs) GridSearchCV models
    rf_best, rf_best_results = rf_models.grid_search_RF(x_train, x_test, y_train, y_test, target, n)
    pickle.dump(rf_best, open(f'{outdir}/random_forest/{target}_{disease}_{network}_rf_best_model.csv', 'wb'))

    # Support Vector Machines (SVMs) GridSearchCV models
    svm_lin_best, svm_lin_best_results = svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, target, n, 'linear')
    svm_rbf_best, svm_rbf_best_results = svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, target, n, 'rbf')
    pickle.dump(svm_lin_best, open(f'{outdir}/svm/{target}_{disease}_{network}_lin_best_model.csv', 'wb'))
    pickle.dump(svm_rbf_best, open(f'{outdir}/svm/{target}_{disease}_{network}_rbf_best_model.csv', 'wb'))

    # Logistic Regression (LogReg)
    logreg_base, logreg_base_results = logreg_models.base_model_LogReg(x_train_scaled, x_test_scaled, y_train, y_test)

    # All models results
    results = [rf_best_results, svm_lin_best_results, svm_rbf_best_results, logreg_base_results]
    results = pd.DataFrame(results, dtype = float)
    results.to_csv(f'{outdir}/{target}_split_CV{n}_results_{disease}_{network}.tsv', sep='\t')

    return results


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
