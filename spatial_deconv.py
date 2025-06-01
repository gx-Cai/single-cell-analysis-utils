# Author: Cgx
# Date: 2023.06.27
# # This file is modified from https://github.com/QuKunLab/SpatialBenchmarking/tree/main
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
from time import ctime

from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text

import cell2location
import scvi
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel


def printv(*args,verbose=True):
    if not verbose: 
        return None
    else:
        print(ctime(), *args)

def run_cell2location(scrna_data, spatial_data, celltype_key, N_cells_per_location=30, use_gpu=True, verbose=True):
    
    scrna_data_raw = scrna_data.copy()
    spatial_data = spatial_data.copy()
    
    printv("Reading data and perform basic filtering.",verbose=verbose)
    spatial_data.var_names_make_unique()
    scrna_data_raw.X = csr_matrix(scrna_data_raw.X)
    spatial_data.X = csr_matrix(spatial_data.X)
    scrna_data_raw = scrna_data_raw[
        ~scrna_data_raw.obs[celltype_key].isin(
            np.array(scrna_data_raw.obs[celltype_key].value_counts()[scrna_data_raw.obs[celltype_key].value_counts() <=1].index))
    ]
    # remove cells and genes with 0 counts everywhere
    sc.pp.filter_genes(scrna_data_raw,min_cells=1)
    sc.pp.filter_cells(scrna_data_raw,min_genes=1)
    scrna_data_raw.obs[celltype_key] = pd.Categorical(scrna_data_raw.obs[celltype_key])
    scrna_data_raw = scrna_data_raw[~scrna_data_raw.obs[celltype_key].isna(), :]
    selected = filter_genes(scrna_data_raw, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    # filter the object
    scrna_data_raw = scrna_data_raw[:, selected].copy()

    printv("Traing regression model", verbose=verbose)
    RegressionModel.setup_anndata(adata=scrna_data_raw,labels_key=celltype_key)
    # create and train the regression model
    mod = RegressionModel(scrna_data_raw)
    # Use all data for training (validation not implemented yet, train_size=1)
    mod.train(max_epochs=300, batch_size=2500, train_size=1, lr=0.002, use_gpu=use_gpu)
    # plot ELBO loss history during training, removing first 20 epochs from the plot
    if verbose:
        mod.plot_history(20)
        plt.show()

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    scrna_data_raw = mod.export_posterior(
        scrna_data_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': use_gpu}
    )

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in scrna_data_raw.varm.keys():
        inf_aver = scrna_data_raw.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in scrna_data_raw.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = scrna_data_raw.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in scrna_data_raw.uns['mod']['factor_names']]].copy()
    inf_aver.columns = scrna_data_raw.uns['mod']['factor_names']    
    intersect = np.intersect1d(spatial_data.var_names, inf_aver.index)
    intersect = np.unique(intersect)
    spatial_data = spatial_data[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    printv("Traing cell2location model", verbose=verbose, )
    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=spatial_data)
    #scvi.data.view_anndata_setup(adata_vis)
    # create and train the model
    mod = cell2location.models.Cell2location(
        spatial_data, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=N_cells_per_location,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection (using default here):
        detection_alpha=200,
    )

    mod.train(
        max_epochs=5000,
        # train using full data (batch_size=None)
        batch_size=None,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1,
        early_stopping=True,
        early_stopping_monitor = 'elbo_train',
        # lr = 0.002,
        use_gpu=use_gpu,
    )
    
    if verbose:
        # plot ELBO loss history during training, removing first 100 epochs from the plot
        mod.plot_history(1000)
        plt.legend(labels=['full data training'])
        plt.show()
        print(spatial_data)
    
    spatial_data = mod.export_posterior(
        spatial_data, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': use_gpu}
    )
    return inf_aver, spatial_data.obsm['q05_cell_abundance_w_sf']


def run_tangram(ad_sc, ad_sp, celltype_key ):
    celltype_counts = ad_sc.obs[celltype_key].value_counts()
    celltype_drop = celltype_counts.index[celltype_counts < 2]
    print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
    ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()

    # remove cells and genes with 0 counts everywhere
    sc.pp.filter_genes(ad_sc,min_cells=1)
    sc.pp.filter_cells(ad_sc,min_genes=1)

    # del rows without celltype
    ad_sc.obs[celltype_key] = pd.Categorical(ad_sc.obs[celltype_key])
    ad_sc = ad_sc[~ad_sc.obs[celltype_key].isna(), :].copy()

    selected, ad_sc, ad_sp = SelectGenes(ad_sc, ad_sp, select_gene_method, celltype_key)

    tg.pp_adatas(ad_sc, ad_sp, genes=selected)

    ad_map = tg.map_cells_to_space(
                       ad_sc,
                       ad_sp,
                       mode='clusters',
                       cluster_label=celltype_key)

    tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)

    celltype_density = ad_sp.obsm['tangram_ct_pred']
    celltype_density = (celltype_density.T/celltype_density.sum(axis=1)).T
    
    return celltype_density