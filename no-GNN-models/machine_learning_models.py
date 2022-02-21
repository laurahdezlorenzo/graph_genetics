import sys
import os
import pandas as pd
import numpy as np

import pickle
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

import svm_models, rf_models, logreg_models

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

    # Select dataset to use
    if DS == 'ADNI': # Use ADNI data
        infile = f'{INDIR}/{DIS}_{NET}_missense_with_biomarkers.csv'
        indata = pd.read_csv(infile, index_col=0)
        data_wclass = create_class_ADNI(indata, TAR)

    # Run parameters
    split = 0.1 # Percentage of test set
    n = 5       # Number of folds for GridSearchCV
    print()
    print(f'Classification with {TAR} - Split {split*100}% - GridSearchCV {n}')

    # Data preprocessing
    x = data_wclass.drop(['y'], axis=1)
    y = data_wclass['y']

    # Load the split used for GNNs models.
    x.reset_index(inplace=True)
    x = x.drop(['index'], axis=1)
    print(x)

    f = open(f'data/split_{TAR}.pkl', 'rb')
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
    rf_best, rf_best_results = rf_models.grid_search_RF(x_train, x_test, y_train, y_test, TAR, n)
    pickle.dump(rf_best, open(f'{OUTDIR}/random_forest/{TAR}_{DIS}_{NET}_rf_best_model.csv', 'wb'))

    # Support Vector Machines (SVMs) GridSearchCV models
    svm_lin_best, svm_lin_best_results = svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, TAR, n, 'linear')
    svm_rbf_best, svm_rbf_best_results = svm_models.grid_search_SVM(x_train_scaled, x_test_scaled, y_train, y_test, TAR, n, 'rbf')
    pickle.dump(svm_lin_best, open(f'{OUTDIR}/svm/{TAR}_{DIS}_{NET}_lin_best_model.csv', 'wb'))
    pickle.dump(svm_rbf_best, open(f'{OUTDIR}/svm/{TAR}_{DIS}_{NET}_rbf_best_model.csv', 'wb'))

    # Logistic Regression (LogReg)
    logreg_base, logreg_base_results = logreg_models.base_model_LogReg(x_train_scaled, x_test_scaled, y_train, y_test)

    # All models results
    results = [rf_best_results, svm_lin_best_results, svm_rbf_best_results, logreg_base_results]
    results = pd.DataFrame(results, dtype = float)
    results.to_csv(f'{OUTDIR}/{TAR}_split_CV{n}_results_{DIS}_{NET}.tsv', sep='\t')

