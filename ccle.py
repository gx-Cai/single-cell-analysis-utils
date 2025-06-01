import pandas as pd
import numpy as np
import os
from anndata import AnnData
import gseapy as gp
from scipy.stats import ranksums
import statsmodels.stats.multitest as ssm
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from . import config

base_dir = os.path.join(config.config.data_dir, 'CCLE')

def get_CellLine_data(target_tcga):
    expr = pd.read_csv(os.path.join(base_dir,'CCLE_expr_22Q2.csv'), index_col=0).T # log(TPM+1)
    GDSC_info = pd.read_excel(os.path.join(base_dir,'Cell_Lines_Details.xlsx'))
    CCLE_info = pd.read_csv(os.path.join(base_dir,'sample_info.csv'))
    GDSC_info.columns = [i.replace(' ','').replace('\n','') for i in GDSC_info.columns]
    ic50 = pd.read_csv(os.path.join(base_dir,'sanger-dose-response.csv'), index_col='COSMIC_ID')

    target_cell_lines = GDSC_info.query('`CancerType(matchingTCGAlabel)` in @target_tcga')
    print(f'Search target cancer type {target_tcga}, find {target_cell_lines.shape[0]} cell lines.')
    print('\t- Trying to matching by cosmic ID.', end='\t')
    target_cosmic = target_cell_lines['COSMICidentifier'].dropna().astype(int).values
    cosmic_id = CCLE_info['COSMICID'].dropna().astype(int)
    sample_idx1 = CCLE_info.loc[cosmic_id[cosmic_id.isin(target_cosmic)].index,'DepMap_ID']
    print(f'\tfind {sample_idx1.size} cell lines.')

    print('\t- Trying to matching by cell line name.', end='\t')
    target_name = target_cell_lines['SampleName']
    sample_idx2 = CCLE_info.loc[CCLE_info[CCLE_info['cell_line_name'].isin(target_name)].index,'DepMap_ID']
    print(f'\tfind {sample_idx2.size} cell lines.')

    sample_idx = pd.Index(sample_idx1).union(pd.Index(sample_idx2))
    sample_info = CCLE_info.set_index('DepMap_ID').loc[sample_idx,:]
    expr = expr.loc[:,sample_info.index.intersection(expr.columns)]
    expr.index = [i.split(' (')[0] for i in expr.index]
    expr = (expr.T - expr.mean(axis=1))/expr.std(axis=1)
    print(f'merging. get {sample_idx.size} cell lines. {expr.shape[0]} cell lins have expression.')

    matched_cosmicids = sample_info['COSMICID'].astype(int)
    matched_cosmicids = matched_cosmicids[matched_cosmicids.isin(ic50.index)]

    ic50_mat = (ic50
    .loc[matched_cosmicids,:]
     .reset_index()
     .pivot_table(values = 'IC50_PUBLISHED', index='DRUG_ID', columns = 'ARXSPAN_ID').T
    )
    data = AnnData(
        expr, obs = sample_info.loc[expr.index,:], 
        obsm = {'ic50': ic50_mat.loc[expr.index,:]},
        uns = {'drug': ic50[['DRUG_ID','DRUG_NAME']].drop_duplicates().dropna().reset_index(drop=True)}
    )
    
    return data #sample_info, expr


def multigenes_test(data, target_genes, weight=None, q=0.5, add_key = 'multigenes_test'):    
    if weight is None:
        tg_score = pd.Series(data[:,target_genes].X.sum(axis=1), index=data.obs_names)
    else:
        assert len(target_genes) == len(weight)
        tg_score = pd.Series(data[:,target_genes].X @ np.array(weight), index=data.obs_names)
    
    high_samples = tg_score.index[tg_score > tg_score.quantile(q)]
    low_samples = tg_score.index[tg_score < tg_score.quantile(1-q)]
    data.obs.loc[high_samples,'multigenes_test_group'] = 'High'
    data.obs.loc[low_samples,'multigenes_test_group'] = 'Low'
    data.obs['multigenes_score'] = tg_score
    ic50_mat = data.obsm['ic50'].T
    high_ic50_median = np.nanmedian(ic50_mat.loc[:, high_samples], axis=1)
    low_ic50_median = np.nanmedian(ic50_mat.loc[:, low_samples], axis=1)
    fc = high_ic50_median / low_ic50_median
    
    drug_pval = ranksums(ic50_mat.loc[:, high_samples].T,ic50_mat.loc[:, low_samples].T, nan_policy = 'omit').pvalue
    rejects, fdrs, _0, _1 = ssm.multipletests(drug_pval, method = 'fdr_bh')

    result = pd.merge(
        left=pd.DataFrame(
            {
                'FoldChange': fc,
                'logFoldChange': np.log2(fc),
                'pval': drug_pval,
                'adj_pval': fdrs
            },
            index = ic50_mat.index
        ), 
        right = data.uns['drug'],
        left_index=True, right_on='DRUG_ID', how='left'
    ).set_index('DRUG_NAME')
    data.uns[add_key] = {
        'target_genes': target_genes,
        'result': result,
    }
    return result # .query('pval < 0.05 & (logFoldChange > 1 | logFoldChange < -1)')


def drug_sensitivity_corr(data, gene_sets:dict, add_key = 'drug_sensitivity_corr'):
    ssg_res = gp.ssgsea(
        data.to_df().dropna(axis=1).T,
        gene_sets = gene_sets
    )

    X_sig = ssg_res.res2d.pivot_table(values='NES', index='Name', columns='Term')
    result = {}
    for i in data.obsm['ic50'].columns:
        x = data.obsm['ic50'].loc[:,i].dropna()
        result[i] = {}
        for gs in gene_sets.keys():
            cor, p = pearsonr(x,X_sig.loc[x.index,gs])
            result[i].update({
                f'{gs}_corr':cor,
                f'{gs}_pval':p
            })
    result = pd.DataFrame(result).T
    
    data.uns[add_key] = {
        'gene_sets':gene_sets,
        'result': result #pd.concat([data.obsm['ic50'], X_sig], axis=1).corr().loc[list(gene_sets.keys())]
    }
    return data.uns[add_key]['result']


def drug_sensetivity_corr_heatmap(target_gene_set, data, ax = None, key = 'drug_sensitivity_corr', **heatmap_kwgs):
    sns.set_style('white')
    drug_sen = data.uns['drug_sensitivity_corr']['result']
    dfvis = drug_sen.query(f'`{target_gene_set}_pval` < 0.05').sort_values(by=f'{target_gene_set}_corr')[[f'{target_gene_set}_corr']].T
    if ax is None:
        f,ax = plt.subplots(dpi=300, figsize=(dfvis.shape[1] // 3,1))
        # ax = plt.gca()
        
    default_params = dict(annot=True, fmt='.2f', ax=ax , lw=1, cmap='coolwarm', annot_kws=dict(rotation=90), center=0, cbar_kws={'label':'pearson r'})
    default_params.update(heatmap_kwgs)
    ax = sns.heatmap(
        dfvis,
        **default_params
    )
    ax.set_title(target_gene_set)
    ax.set_yticklabels([])
    drug_name = ax.set_xticklabels(
        [data.uns['drug'].set_index('DRUG_ID')['DRUG_NAME'].to_dict().get(drug_id:=int(i.get_text()),drug_id) for i in ax.get_xticklabels()]
    )
    return ax


def drug_ic50_group_barplot(target_drugs:pd.DataFrame, drug_data, yi, ax=None):
    from .visulize_utils import annote_line
    
    dfvis = drug_data.obsm['ic50'].loc[:, target_drugs['DRUG_ID']].reset_index().melt(id_vars = ['index'], value_vars = target_drugs['DRUG_ID']).set_index('index')
    dfvis = pd.merge(dfvis, drug_data.obs[['multigenes_test_group']], left_index=True, right_index=True)
    dfvis.loc[:,'DRUG_NAME'] = dfvis['DRUG_ID'].map({v:k for k,v in target_drugs['DRUG_ID'].to_dict().items()})

    ax = sns.barplot(
        dfvis.reset_index(drop=True), x = 'DRUG_NAME', y = 'value', hue = 'multigenes_test_group',
        palette = 'Set2',errwidth=1.5, #fliersize=3,
        ax=ax
    )
    ax.legend(title='',frameon=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('IC50')

    for it in ax.get_xticklabels():
        g = it.get_text()
        p = target_drugs.loc[g, 'pval']
        xi,_ = it.get_position()
        # yi = -8
        if p < 1e-3:
            annote = '***'
        elif p < 1e-2:
            annote = '**'
        elif p < 0.05:
            annote = '*'
        else:
            annote = ''
        annote_line(
            ax, s=annote, start=(xi-0.5, yi), end=(xi+0.5, yi), lw=0, color='.4', ygap=0
        )
    return ax