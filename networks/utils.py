import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mne.stats import fdr_correction, bonferroni_correction
import matplotlib.pyplot as plt
import seaborn as sns

def boxplot_comparision_others(target, df, metric, title):
    
    order=['Only APOE', 'AD BioGRID', 'AD GIANT', 'AD HuRI', 'AD PPT-Ohmnet', 'AD STRING']
    colors = ["#F8766D", "#E76BF3", "#E76BF3", "#E76BF3", "#E76BF3", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x = 'dataset', y = metric, data = df, palette = custom, order=order)
                    #  ci = 'sd',

    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=16, rotation=40)
    plt.yticks(fontsize=16)
#     ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric.upper()}', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
#     plt.show()
    
#     plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
    plt.savefig(f'figures/figure2{title}.png', dpi=500)

def violinplot_comparision_others(target, df, metric, title):
    
    colors = ["#E76BF3", "#9590ff", "#9590ff", "#9590ff", "#9590ff"]
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
    plt.show()
    
    # plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
    # plt.savefig(f'figures/figure3{title}.png', dpi=500)

def barplot_comparision_others(target, df, metric, title):
    
    colors = ["#E76BF3", "#9590ff", "#9590ff", "#9590ff", "#9590ff"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x = 'dataset', y = metric, data = df, palette = custom, ci = 'sd')

    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=16, rotation=40)
    plt.yticks(fontsize=16)
    ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric} obtained in test set', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
    # plt.savefig(f'figures/figure3{title}.png', dpi=500)


def statistics(df):
    pvalues = {}
    for d in df['dataset'].unique():
        tmp = df.loc[df['dataset'] == d]['auc'].values
        bas = df.loc[df['model'] == 'Baseline model']['auc'].values
        t, pval = stats.ttest_ind(tmp, bas, alternative='greater')
        pvalues[d] = pval
    
    pvalues_sorted = {k: v for k, v in sorted(pvalues.items(), key=lambda item: item[1])}
    
    print('Against baseline:')
    for k in pvalues_sorted:
        p = pvalues_sorted[k]
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), k)
        else:
            print('( )', '{:0.4e}'.format(p), k)
    print()

    return pvalues_sorted


def pvalues_random(original_results, random_results, random_method):
    
    '''Compute p-values comparing original AUC values obtained with PPI graph datasets
    vs. AUC values obtained with different random networks'''
    
    gnn_results = original_results.loc[original_results['dataset'] == 'AD PPT-Ohmnet']
    original_aucs = list(gnn_results['auc'].values)

    
    random_aucs = []
    for i in range(100):
        i += 1
        net = f'AD_PPI_{random_method}{i}_missense'
        tmp_results = random_results.loc[random_results['dataset'] == net]
        tmp_aucs = list(tmp_results['auc'].values)
        random_aucs.append(tmp_aucs)

    ts, pvals = stats.ttest_1samp(random_aucs, original_aucs, alternative='less') # 1 sample t-test each fold
    reject_fdr, pvals_fdr = fdr_correction(pvals, alpha=0.05, method='indep') # Benjamini-Hochberg correction

    print(f'{random_method} vs. original AUCs:')
    k = 1
    for p in pvals_fdr:
        if p < 0.05:
            print('(*)', '{:0.4e}'.format(p), 'Fold', k)
        else:
            print('( )', '{:0.4e}'.format(p), 'Fold', k)

        k +=1
    print()
    
    return pvals_fdr


