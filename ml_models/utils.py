import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def boxplot_comparison_models(target, df, metric, title):
    
    if title == 'LOAD':
        dat_order=['Only APOE', 'AD PPT-Ohmnet']
    else:
        dat_order=['Only APOE', 'AD PPT-Ohmnet','AD PPT-Ohmnet no APOE']
        
    hue_order=['Baseline model', 'Logistic Regression','SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']
    colors = ["#F8766D", "#a3a500", "#00bf7d", "#00b0f6", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x='dataset', hue='model', y=metric, data=df, palette=custom, order=dat_order, hue_order=hue_order)

    plt.ylim(0.3, 1.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
#     ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric.upper()}', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
#     plt.show()
    
    # plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
#     plt.savefig(f'figures/figure3{title}.png', dpi=500)

def violinplot_comparison_models(target, df, metric, title):
    
    colors = ["#F8766D", "#a3a500", "#00bf7d", "#00b0f6", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x = 'dataset', y = metric, data = df, palette = custom)
    ax = sns.swarmplot(x = 'dataset', y = metric, data = df, palette = custom, color='white')

    plt.ylim(0.2, 1.0)
    plt.xticks(fontsize=16, rotation=40)
    plt.yticks(fontsize=16)
    ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric} obtained in test set', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
#     plt.show()
    
    # plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
    # plt.savefig(f'figures/figure3{title}.png', dpi=500)

def barplot_comparison_models(target, df, metric, title):
    
    if title == 'LOAD':
        dat_order=['Only APOE', 'AD PPT-Ohmnet']
    else:
        dat_order=['Only APOE', 'AD PPT-Ohmnet','AD PPT-Ohmnet no APOE']
        
    hue_order=['Baseline model', 'Logistic Regression','SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']
    colors = ["#F8766D", "#a3a500", "#00bf7d", "#00b0f6", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x='dataset', hue='model', y=metric, data=df, palette=custom, order=dat_order, ci='sd',hue_order=hue_order )

    plt.ylim(0.3, 1.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
#     ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric.upper()}', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # plt.savefig(f'figures/figure3{title}_barplot.pdf', dpi=500)
    plt.savefig(f'figures/figure3{title}_barplot.png', dpi=500)


def statistics(df):

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
