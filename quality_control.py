import scanpy as sc
import pandas as pd
import numpy as np
import os
from scipy.stats import median_abs_deviation
import anndata2ri
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
from . import config

__all__ = ['filtering_reads','filtering_drop']

RHOME = config.R_home_path

RLIB = os.path.join(RHOME,'library')
file_path = os.path.dirname(os.path.abspath(__file__))

r = ro.r
r(f'Sys.setenv(R_HOME = \"{RHOME}\")')
r(f'.libPaths(\"{RLIB}\")')
r.source(os.path.join(file_path,'./quality_control.R'))

rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()

sc.settings.verbosity = 0

def filtering_reads(adata, verbose = True):

    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier
    adata = adata.copy()
    adata.var_names_make_unique()
    raw_varinfo = adata.var_keys()
    raw_obsinfo = adata.obs_keys()
    
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", 5)
        | is_outlier(adata, "log1p_n_genes_by_counts", 5)
        | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
        adata.obs["pct_counts_mt"] > 8
    )
    
    raw_nobs = adata.n_obs
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    
    if verbose:
        print(f"Total number of cells: {raw_nobs}")
        print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")
    
    qc_varkeys = list(set(adata.var_keys()) - set(raw_varinfo))
    qc_obskeys = list(set(adata.obs_keys()) - set(raw_obsinfo))
    
    adata.uns['qc'] = {
        'obs': adata.obs[qc_obskeys],
        'var': adata.var[qc_varkeys]
    }
    
    adata.obs = adata.obs[raw_obsinfo]
    adata.var = adata.var[raw_varinfo]
    
    return adata

def filtering_drop(adata, adata_raw, verbose=True):
    
    def leiden_clustering(adata):
        adata_pp = adata.copy()
        sc.pp.normalize_per_cell(adata_pp)
        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp)
        sc.pp.neighbors(adata_pp)
        sc.tl.leiden(adata_pp, key_added="groups")
        groups = adata_pp.obs["groups"]
        return groups

    cells = adata.obs_names
    genes = adata.var_names
    data = adata.X.T
    data_tod = adata_raw.X.T
    if verbose:
        print('running leigen cluster for soupX inputs ...')
    soupx_groups = leiden_clustering(adata)
    
    if verbose:
        print('running soupX Correction ...')
        
    out = r['run_soupX'](data, data_tod, genes, cells, soupx_groups)
    adata.layers["counts"] = adata.X
    adata.layers["soupX_counts"] = out.T
    if verbose:
        print('saving result in layers as soupX_counts and replacing the X')
    adata.X = adata.layers["soupX_counts"]
    sc.pp.filter_genes(adata, min_cells=20)
    
    if verbose:
        print('running scdblfinder for doublet finding. But will not filerting.')
        print('saving the result into .obs doublet_score and doublet_class')
        
    scdb_res = r['run_scdblfinder'](data)
    adata.obs['doublet_score'] = scdb_res[0]
    adata.obs['doublet_class'] = pd.Categorical(scdb_res[1])
