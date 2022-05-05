import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def pvalues_nonGNNs(df):

    pvalues_bas = {}
    pvalues_gnn = {}
    pvalues_ran = {}

    bas = df.loc[df['model'] == 'Baseline model']['auc'].values
    
    ran = [0.5]*10

    for d in ['AD PPT-Ohmnet', 'AD PPT-Ohmnet no APOE']: 
        for m in ['Logistic Regression', 'SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']: # when GG results no APOE
            
            tmp = df.loc[(df['dataset'] == d) & (df['model'] == m)]['auc'].values
            gnn = df.loc[(df['dataset'] == d) & (df['model'] == 'GNN GraphGym')]['auc'].values
#             print(d, m)
#             print(tmp)
#             print(gnn)
#             print()
            t_bas, pval_bas = stats.ttest_ind(tmp, bas, alternative='greater')
            t_gnn, pval_gnn = stats.ttest_ind(tmp, gnn, alternative='less')
            t_ran, pval_ran = stats.ttest_ind(tmp, ran, alternative='greater')
            
            pvalues_bas[f'{d} - {m}'] = pval_bas
            pvalues_gnn[f'{d} - {m}'] = pval_gnn
            pvalues_ran[f'{d} - {m}'] = pval_ran

        pvalues_bas_sorted = {k: v for k, v in sorted(pvalues_bas.items(), key=lambda item: item[1])}
        pvalues_gnn_sorted = {k: v for k, v in sorted(pvalues_gnn.items(), key=lambda item: item[1])}
        pvalues_ran_sorted = {k: v for k, v in sorted(pvalues_ran.items(), key=lambda item: item[1])}

    print('Against baseline:')
    for k in pvalues_bas_sorted:
        p = pvalues_bas_sorted[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    print()
    
    print('Against GNN:')
    for k in pvalues_gnn_sorted:
        p = pvalues_gnn_sorted[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    print()

    print('Against random:')
    for k in pvalues_ran_sorted:
        p = pvalues_ran_sorted[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
            
    return pvalues_bas, pvalues_ran
