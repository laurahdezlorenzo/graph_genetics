import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def pvalues_nonGNNs(df):

    pvalues_bas = {}
    pvalues_gnn = {}
    pvalues_ran = {}
    pvalues_apoe = {}

    bas = df.loc[df['model'] == 'Baseline model']['auc'].values

    for d in ['AD PPT-Ohmnet', 'AD PPT-Ohmnet no APOE']:        
        for m in ['Logistic Regression', 'SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']:
            
            tmp = df.loc[(df['dataset'] == d) & (df['model'] == m)]['auc'].values
            gnn = df.loc[(df['dataset'] == d) & (df['model'] == 'GNN GraphGym')]['auc'].values
#             print(d, m)
#             print(tmp.mean())
#             print(gnn.mean())
#             print()
            t_bas, pval_bas = stats.ttest_ind(tmp, bas, alternative='greater')
            t_gnn, pval_gnn = stats.ttest_ind(tmp, gnn, alternative='less')
            t_ran, pval_ran = stats.ttest_1samp(tmp, 0.5, alternative='less')
            
            pvalues_bas[f'{d} - {m}'] = pval_bas
            pvalues_gnn[f'{d} - {m}'] = pval_gnn
            pvalues_ran[f'{d} - {m}'] = pval_ran
    
    for m in ['Logistic Regression', 'SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']:
        
        apoe    = df.loc[(df['dataset'] == 'AD PPT-Ohmnet')  & (df['model'] == m)]['auc'].values
        no_apoe = df.loc[(df['dataset'] == 'AD PPT-Ohmnet no APOE')  & (df['model'] == m)]['auc'].values
        
        t_apoe, pval_apoe = stats.ttest_ind(no_apoe, apoe, alternative='less')
        pvalues_apoe[f'{m}'] = pval_apoe
        

    print('Against baseline:')
    for k in pvalues_bas:
        p = pvalues_bas[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    print()
    
    print('Against GNN:')
    for k in pvalues_gnn:
        p = pvalues_gnn[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    print()

    print('Against random:')
    for k in pvalues_ran:
        p = pvalues_ran[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    
    print()
    
    print('With APOE vs. without APOE:')
    for k in pvalues_apoe:
        p = pvalues_apoe[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
            
    return pvalues_bas, pvalues_ran
