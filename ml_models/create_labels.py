import sys, os
import pandas as pd
import numpy as np

def labeling(row, option):
        
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

def label_ADNI(df, option):

    
    df['y'] = df.apply (lambda row: labeling(row, option), axis=1)

    to_drop = list(df.columns)[-27:-1]
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    # print('Class distribution:')
    # print(df['y'].value_counts())

    return df_notna

def label_LOAD(df):

    
    df['y'] = df['Phenotype']
    to_drop = ['FID', 'father_IID', 'mother_IID', 'Sex', 'Phenotype']
    df.drop(columns=to_drop, inplace=True)
    df_notna = df[df['y'].notna()]

    df_notna['y'].replace({2:1, 1:0}, inplace=True)

    # print('Class distribution:')
    # print(df_notna['y'].value_counts())

    return df_notna