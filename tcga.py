import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .survival import *
from . import config
import os


tcga_data_dir = config.data_dir
data_dir = os.path.join(tcga_data_dir, 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz')
os_dir = os.path.join(tcga_data_dir, 'Survival_SupplementalTable_S1_20171025_xena_sp.txt')
type_dir = os.path.join(tcga_data_dir, 'TCGASubtype.20170308.tsv.gz')

data = pd.read_table(data_dir,encoding='utf-8',index_col=0)
data = np.power(2,data) - 1
data = 1e6* data / data.sum(axis=0)
sample_type = pd.read_table(type_dir)
dataos = pd.read_table(os_dir)

valid_idx = data.columns.str[0:12].isin(dataos['_PATIENT'])
data = data.loc[:,valid_idx]
sample_map = dataos.loc[:,['_PATIENT','cancer type abbreviation']].drop_duplicates().set_index('_PATIENT',drop=True)
all_samples = pd.DataFrame(columns = ['sample','site'],index= data.columns)
all_samples['sample'] = all_samples.index.str[0:12]
all_samples['site'] = all_samples.index.str[-2:]
all_samples = pd.merge(left = all_samples,right = sample_map,right_index=True,left_on='sample')

add_clinical = pd.read_table('/share/home/biopharm/Caiguoxin/Data/TCGA/NSCLC/combined_study_clinical_data.tsv', index_col='Patient ID')
copd_samples = add_clinical.index[add_clinical['Fev1 percent ref postbroncholiator'] <= 70] +'-01'
noncopd_samples = add_clinical.index[add_clinical['Fev1 percent ref postbroncholiator'] > 70] +'-01'

def load_cancer_expr(
    cancer_type, # str or list
    survival_type = 'OS', # OS DSS or PFI DFI
    extra_columns = []
):
    if type(cancer_type) is str:
        samples = all_samples[all_samples['cancer type abbreviation'] == cancer_type]
    elif type(cancer_type) is list:
        samples  = all_samples.loc[all_samples.query("`cancer type abbreviation` in @cancer_type").index,:]
    df = data.loc[:,samples.index]
    df = df.fillna(0)
    df = df[~(df == 0).all(axis=1)]
    idx = ~df.index.duplicated()
    df = df[idx]
    
    survival_cols = dataos.filter(like = survival_type).columns.tolist()
    samples_os = dataos.set_index('_PATIENT',drop=True).loc[
        pd.Index(samples['sample'].values).intersection(
            pd.Index(samples['sample'].values)),
        survival_cols + ['cancer type abbreviation'] + extra_columns
    ]
    samples_os.index += '-01'
    df = (df.T - df.mean(axis=1)) / df.std(axis=1)
    df = pd.merge(df, samples_os, left_index=True,right_index=True)
    df = df.rename(
        columns={
            i:i.replace(survival_type,'OS')
            for i in survival_cols
        }
    )
    return df

def searching_genes_plot(
    df, target_genes:list,
    # figure args. 
    figargs={},
    nrow=None,ncol=None,
    scale=1.5, figshape = None,
    **extra_QSSP_args
):
    target_genes =  list(set(target_genes).intersection(df.columns))
    if (nrow is None) or (ncol is None):
        if (n_genes:= len(target_genes)) < 10:
            nrow = 2
        else:
            nrow = n_genes // 5 + 1

        ncol = int(n_genes / nrow)
        if n_genes % nrow !=0: ncol += 1
    print('Overlaped genes:', len(target_genes), f'setting figure <{nrow};{ncol}>')
    if figshape is None:
        figshape = (nrow, ncol)
    f,axes = plt.subplots(nrow,ncol,dpi=300,figsize=(figshape[0]*scale,figshape[1]*scale), **figargs)
    # plt.subplots_adjust(hspace=0.4)
    axes = axes.flatten() if ((nrow > 1) & (ncol > 1)) else axes
    for gene,ax in zip(target_genes,axes):
        best_p = 1
        best_q = 1
        for q in np.linspace(0.1,0.5,10):
            p = QuantileSampleSplit(df,gene = gene, q=q, verbose=False)
            if p < best_p:
                best_p = p
                best_q = q
        # print(gene, best_p, best_q)
        QuantileSampleSplitPlot(df.dropna(),gene=gene, q=best_q,ax=ax, **extra_QSSP_args)
    return f, axes

def expr_grouping_plot(target_genes, ax=None):
    from scipy.stats import ranksums
    from utils.visulize_utils import annote_line
    
    if ax is None:
        ax = plt.gca()
    
    target_genes =  list(set(target_genes).intersection(data.index))
    print('Valid genes:')
    print(*target_genes)
    dfvis = pd.concat([
        np.log(data.loc[target_genes, copd_samples.intersection(data.columns)].T),
        np.log(data.loc[target_genes, noncopd_samples.intersection(data.columns)].T)],
        keys= ['COPD','nonCOPD']
    ).reset_index()
    dfvis = dfvis.melt(id_vars=['level_0','level_1'], value_vars=target_genes, value_name='gene_expr')
    ax = sns.barplot(
        dfvis, y = 'gene_expr', x='sample', hue ='level_0',
        width=0.8, fill=True, palette='Set2', err_kws=dict(lw=1), edgecolor='.4',
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('log gene expression')
    ax.legend(frameon=False, title='',ncols=2)
    annote_x = []
    for it in ax.get_xticklabels():
        g = it.get_text()
        p = ranksums(
            *dfvis.query('sample == @g').groupby(['level_0']).apply(lambda x: x['gene_expr'].tolist())
        ).pvalue
        # xi,yi = it.get_position()
        # x = dfvis.query('sample == @g')['gene_expr']
        # yi = - ymax / 30
        if p < 1e-3:
            annote = '***'
        elif p < 1e-2:
            annote = '**'
        elif p < 0.05:
            annote = '*'
        else:
            annote = ''
        annote_x.append(g+annote)
        # annote_line(
        #     ax, s=annote, start=(xi-0.5, yi), end=(xi+0.5, yi), lw=0, color='.4', ygap=0
        # )
    ax.set_xticklabels(annote_x)
    return ax