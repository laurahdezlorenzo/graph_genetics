import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def boxplot_comparison_models(target, df, metric, title):
    
    hue_order=['Logistic Regression - Only APOE', 'Logistic Regression',
               'SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']
    colors = ["#F8766D", "#a3a500", "#00bf7d", "#00b0f6", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x = 'dataset', hue='model', y = metric, data = df, palette = custom, hue_order=hue_order)

    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=16)
    ax.xaxis.label.set_visible(False)
    plt.ylabel(f'{metric.upper()}', fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
#     plt.show()
    
    # plt.savefig(f'figures/figure3{title}.pdf', dpi=500)
    plt.savefig(f'figures/figure3{title}.png', dpi=500)

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
    
    hue_order=['Logistic Regression - Only APOE', 'Logistic Regression',
               'SVM Linear', 'SVM RBF', 'Random Forest', 'GNN GraphGym']
    
    colors = ["#F8766D", "#a3a500", "#00bf7d", "#00b0f6", "#E76BF3"]
    custom = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x = 'dataset', hue='model', y = metric, data = df, palette = custom, ci='sd', hue_order=hue_order )

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
        bas = df.loc[df['model'] == 'Logistic Regression']['auc'].values
        t, pval = stats.ttest_ind(tmp, bas, alternative='greater')
        pvalues[d] = pval
    
    pvalues_sorted = {k: v for k, v in sorted(pvalues.items(), key=lambda item: item[1])}

    return pvalues_sorted
