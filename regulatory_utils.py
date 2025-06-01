import warnings
warnings.filterwarnings("ignore")
import shlex
import pyscenic
import loompy as lp
import scanpy as sc
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import json
import zlib
import base64
from pyscenic.rss import regulon_specificity_scores
from pyscenic.utils import modules_from_adjacencies
from matplotlib.colors import to_hex
import networkx as nx
import seaborn as sns
import glob
import os

from . import config

pyscenic_path = config.config.pyscenic_path

db_glob = None
motif_path = None
tfs_path = None

for _, root, files in os.walk(pyscenic_path):
    for file in files:
        if file.endswith('.feather'):
            db_glob = os.path.join(root, '*feather')
        elif file.endswith('.tbl'):
            motif_path = os.path.join(root, file)
        elif file.endswith('allTFs_hg38.txt'):
            tfs_path = os.path.join(root, file)
            tfs = [tf.strip() for tf in open(tfs_path)]
            print(f'Found {len(tfs)} TFs in {tfs_path}')
            if len(tfs) == 0:
                raise ValueError(f'No TFs found in {tfs_path}. Please check the file.')

if db_glob is None or motif_path is None or tfs_path is None:
    raise ValueError(
        f'Please check the pyscenic_path: {pyscenic_path}. '
        'It should contain at least one feather file, one tbl file, and the allTFs_hg38.txt file.'
        'Check the config of pyscenic_path or set it by config.pyscenic_path = "/path/to/pyscenic"'
    )

db_names = " ".join(glob.glob(db_glob))
tfs = [tf.strip() for tf in open(tfs_path)]

sns.set_style('white')

def extract_aucmat(scenic_outs_dir):
    data = sc.read_h5ad(scenic_outs_dir)
    auc_mat = pd.DataFrame(
        data.obsm['X_aucell'],
        columns = data.uns['aucell']['regulon_names'],
        index = data.obs_names
    )
    return auc_mat

def build_graph_from_scenic(
    scenic_out_path, edge_color = 'k',
    selected_TFs=None, selected_context = 'activating_top5perTarget', n_auto_select_tf=15,
    mini_edge_width = 0.5, edge_weight_scale = 2,
    genes_node_size = 10, tf_node_size=200,genes_node_color = 0, tf_node_color = 0.99,
    cmap = 'Set2'
):
    platte = plt.get_cmap(cmap)
    
    outpath_adj = os.path.join(scenic_out_path,'adj.csv')
    loom_path_output = os.path.join(scenic_out_path,"processed_output.loom")
    out_path_reg = os.path.join(scenic_out_path,'reg.csv')
    
    adjacencies = pd.read_csv(outpath_adj, index_col=False, sep=',')
    lf = lp.connect(loom_path_output,mode='r',validate=False)
    exprMat = pd.DataFrame(lf[:,:], index=lf.ra.Gene, columns=lf.ca.CellID).T
    modules = list(modules_from_adjacencies(adjacencies, exprMat))
    if selected_TFs is None:
        print('Auto selected tfs based on the AUC.')
        reg = pd.read_csv(out_path_reg, skiprows=[0,2], index_col=0)
        selected_TFs = reg.sort_values('AUC', ascending=False).index[0:n_auto_select_tf]
        # adjacencies.groupby('TF').apply(lambda x: x['importance'].sum()).sort_values().index[-5:]
    
    network_df =  pd.DataFrame()
    for tf in selected_TFs:
        tf_mods = [ x for x in modules if x.transcription_factor in (tf) ]
        for module in tf_mods:
            df = pd.DataFrame({'Gene':list(module.gene2weight.keys()),'Weight':list(module.gene2weight.values())})
            df[['TF','context']]=module.transcription_factor,'_'.join(sorted(module.context))
            network_df = pd.concat([network_df, df], ignore_index=True)
            
    print(f'Avariable contex: (select {selected_context})')
    print(network_df['context'].value_counts().to_frame().T)
    
    network_df['Weight'] = mini_edge_width+edge_weight_scale*(network_df['Weight'] - network_df['Weight'].min()) / (network_df['Weight'].max() - network_df['Weight'].min())
    network_df['Color'] = edge_color
    
    network_full = network_df.copy()
    # network_df = network_df.groupby('TF').apply(lambda x: x.sort_values('Weight').iloc[-n_downstream_genes:,:]).reset_index(drop=True)
    
    network_df = network_df.query('context == @selected_context')
    G = nx.from_pandas_edgelist(network_df,'TF','Gene',['Weight','Color']) #, 
    Node_df = pd.concat(
        [pd.DataFrame('Gene',index=list(set(network_df['Gene'])),columns=['Node_type']),pd.DataFrame('TF',index=list(set(network_df['TF'])),columns=['Node_type'])],
        axis=0
    )
    Node_df['size'] = np.where(Node_df['Node_type']=='Gene',genes_node_size,tf_node_size)
    Node_df['color'] = np.where(Node_df['Node_type']=='Gene',to_hex(platte(genes_node_color)),to_hex(platte(tf_node_color)))
    node_attrs = Node_df.to_dict().keys()
    for node_attr in node_attrs:
        nx.set_node_attributes(G,Node_df.to_dict()[node_attr],name=node_attr)
    return G, network_full
    
def run_pyscenic(
    data,
    output_path,
    num_workers = 10,
    use_hvg = True,
):
    # parmas extends.
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    outpath_adj = os.path.join(output_path,'adj.csv')
    loom_path = os.path.join(output_path,"processed_input.h5ad")
    loom_path_output = os.path.join(output_path,"processed_output.h5ad")
    loom_integer_output = os.path.join(output_path,"loom_integer_output.loom")
    out_path_reg = os.path.join(output_path,'reg.csv')

    # GOLBAL PARAMS

    if use_hvg:
        if 'highly_variable' not in data.var.columns:
            sc.pp.highly_variable_genes(data, flavor='seurat_v3', n_top_genes=2000, layer = 'counts')

        mask = (data.var["highly_variable"] == True) | data.var.index.isin(tfs)
        data = data[:, mask]
    #as a general QC. We inspect that our object has transcription factors listed in our main annotations.
    print(f"%{np.sum(data.var.index.isin(tfs))} out of {len(tfs)} TFs are found in the object")
    row_attributes = {"Gene": np.array(data.var.index),}
    col_attributes = {
        "CellID": np.array(data.obs.index),
        "nGene": np.array(np.sum(data.X.transpose() > 0, axis=0)).flatten(),
        "nUMI": np.array(np.sum(data.X.transpose(), axis=0)).flatten(),
    }
    if not os.path.exists(os.path.dirname(loom_path)):
        os.mkdir(os.path.dirname(loom_path))
    # lp.create(loom_path, data.X.transpose(), row_attributes, col_attributes)
    data.write_h5ad(loom_path)

    if not os.path.exists(outpath_adj):
        os.system(f"pyscenic grn {shlex.quote(loom_path)} {shlex.quote(tfs_path)} -o {shlex.quote(outpath_adj)} --num_workers {num_workers} --method grnboost2")

    results_adjacencies = pd.read_csv(outpath_adj, index_col=False, sep=",")
    print(f"Number of associations: {results_adjacencies.shape[0]}")

    if not os.path.exists(out_path_reg):
        os.system(f"""pyscenic ctx {shlex.quote(outpath_adj)} \
            {db_names} \
            --annotations_fname {motif_path} \
            --expression_mtx_fname {shlex.quote(loom_path)} \
            --output {shlex.quote(out_path_reg)} \
            --num_workers {num_workers}""")

    if not os.path.exists(loom_path_output):
        os.system(f"""pyscenic aucell {loom_path} \
            {out_path_reg} \
            --output {loom_path_output} \
            --sparse
            # --num_workers {num_workers}""")