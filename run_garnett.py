__all__ = ['run_garnett']

import scanpy as sc
import pandas as pd
import numpy as np
import os
import anndata2ri
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
from scipy.sparse import csc_matrix
import json
from . import config

RHOME = config.R_home_path
RLIB = os.path.join(RHOME,'library')

r = ro.r
r(f'Sys.setenv(R_HOME = \"{RHOME}\")')
r(f'.libPaths(\"{RLIB}\")')
r.source('./utils/run_garnett.R')

rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()


def sizeFactors_sparse(counts):
    """
    median-geometric-mean implement.
    """
    counts = counts[~np.array(((counts !=0).sum(axis=1) ==0)).flatten(),:]
    logcnt = counts.copy()
    logcnt.data = np.log(counts.data)
    # calculate the geometric means along columns
    loggeomeans_vector = lambda i:logcnt.getrow(i).data.mean()
    loggeomeans = np.array([loggeomeans_vector(i) for i in range(counts.shape[0])])
    # calculate the logratios
    logratios = logcnt - loggeomeans[:,None]
    return np.array(np.exp(np.median(logratios,axis=0))).flatten()

def from_dict2marker_file(markers,file_path):
    markers = { k:{kk:vv for kk,vv in v.items()} for k,v in markers.items() if len(v['expressed'])>0 }
    f = open(file_path, 'w')
    for ct, items in markers.items():
        f.write(f'>{ct}\n')
        for k,v in items.items():
            if type(v) == list:
                if len(v) == 0: raise
                s = ', '.join(v)
            elif type(v) == str:
                s = v
            else:
                raise ValueError
            f.write(f'{k}: {s}\n')
        f.write('\n')
    f.close()
    
def run_garnett(
    data, markers, saving_path, 
    variance_thres=False, n_training=None,
    force = True, strict= False, fast_sf = True, 
    verbose=True
):
    def filtering_genes(v,valids):
        if type(v) == str : return v
        elif type(v) == list:
            return [i for i in v if i in valids]
        else:
            raise ValueError
    
    if not os.path.exists(saving_path): os.mkdir(saving_path)
    marker_path = os.path.join(saving_path, 'markers.txt')
    if type(markers) == str: markers = json.load(open(markers))
    
    if 'counts' in data.layers.keys():
        X = data.layers['counts']
    elif hasattr(data,'raw'):
        X = data.raw.X
        data.layers['counts'] = X
    else:
        X = data.X
        data.layers['counts'] = X
    
    X = csc_matrix(X)
    testx = X.data[0:100]
    assert (testx.astype(int)==testx).all(), 'raw counts is not interger'
    
    all_genes = [j for i in markers.values() for j in i.values() if type(j) == list]
    all_genes = data.var_names.intersection([j for i in all_genes for j in i])
    valid_genes = all_genes.copy()
    # variance threshing...
    if (variance_thres > 0) and force:
        if verbose: print('estimate gene variance...')
        sc.pp.highly_variable_genes(data, layer = 'counts', flavor = 'seurat_v3', n_top_genes = 2000)
        thres = data.var['variances'].quantile(q = variance_thres)
        valid_genes = data.var.query("variances > @thres").index.intersection(all_genes)
        if verbose: print('\t',f"{len(valid_genes)} / {len(all_genes)}",'genes have a high variance')
    
    # data = data[:,valid_genes].copy()
    
    # writting marker file
    if force: markers = { k:{kk: filtering_genes(vv, valid_genes) for kk,vv in v.items()} for k,v in markers.items()}
    if verbose: print(f'writting marker file at {marker_path}')
    from_dict2marker_file(markers, marker_path)
    
    # running garnett
    if verbose: print('setting up')
    X = csc_matrix(X.T)
    
    if fast_sf:
        cell_total = X.sum(axis=0)
        sf = np.array(cell_total / np.exp(np.log(cell_total).mean())).flatten()
        sf[np.isnan(sf)] = 1
        data.obs['Size_Factor'] = sf
    
    fdata = data.var
    pdata = data.obs
    flag_precomputed = 'Size_Factor' in pdata.columns
    
    n_training = n_training if n_training is not None else max(data.shape[0] // 20,5000)
    
    cds = r['garnett_setting_up'](X,pdata,fdata, not flag_precomputed)
    if not flag_precomputed:
        estimate_df = ro.pandas2ri.rpy2py_dataframe(r['pData'](cds))
        estimate_df[['Size_Factor']].to_csv(os.path.join(saving_path,'estimate_size_factor.csv'))
    
    if verbose: print('garnett checking marker...')
    marker_check = r['garnett_checking_marker'](cds, marker_path, saving_path)
    marker_check = ro.pandas2ri.rpy2py_dataframe(marker_check)
    if strict: marker_check = marker_check.query('summary == "Ok"')
    else: marker_check = marker_check.query('ambiguity < 0.5')
    if verbose: print('\t',marker_check.shape[0],'genes pass the check')
    if marker_check.shape[0] == 0: raise
    
    if force:
        for k,v in markers.items():
            valid_genes_sub = valid_genes.intersection(marker_check.query('cell_type == @k')['marker_gene'])
            markers[k] = {kk: filtering_genes(vv, valid_genes_sub) for kk,vv in v.items()}
            
    from_dict2marker_file(markers, marker_path)
    
    if verbose: print('running garnett...')
    label = r['run_garnett'](cds, marker_path, saving_path, num_unknown=n_training)
    return ro.pandas2ri.rpy2py_dataframe(label)