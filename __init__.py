from IPython.core.display import display, HTML
import base64
import os
import matplotlib.pyplot as plt
import pandas as pd
from _config import DefaultConfig
__all__ = ['make_toc','update_sample_info','embed_dataframe','embed_multi_objs','embed_multi_objs_folder','export_notebook','saving_fig']

config = DefaultConfig()

def update_sample_info(data, source = 'PSC8', to = 'NSC7'):
    data.obs['sample_id'] = data.obs['sample_id'].astype(str)
    idx = data.obs['sample_id'].str.startswith(source)
    data.obs.loc[idx,'sample_id'] = to + '_' + data.obs.loc[idx,'tissue_type'].astype(str)
    data.obs.loc[idx,'copd'] = 'non-COPD' if to[0] == 'N' else 'COPD'
    data.obs.loc[idx,'disease'] = data.obs.loc[idx,'sample_id'].str[0:3]
    data.obs.index = [i.replace(source,to) if idx[i] else i for i in data.obs.index]
    
def embed_dataframe(df, button='Download the data',filename = 'dataframe.csv', **export_args):
    # encode the dataframe as a csv file
    data = base64.b64encode(df.to_csv(**export_args).encode("utf8")).decode("utf8")

    # create a download link with filename and extension
    link = f'''
    <a href="data:text/csv;base64,{data}" download="{filename}">
    {button}
    </a>
    '''
    # display the link
    display(HTML(link))

def embed_multi_objs(label,**objs):
    import io
    import zipfile
    import pickle

    zip_file = io.BytesIO()
    with zipfile.ZipFile(zip_file, 'w') as zf:
        for k, obj in objs.items():
            if isinstance(obj, pd.DataFrame):
                zf.writestr(f'{k}.csv', obj.to_csv(index=False))
            elif isinstance(obj, pd.Series):
                zf.writestr(f'{k}.csv', obj.to_csv())
            elif isinstance(obj, plt.Figure):
                buf = io.BytesIO()
                obj.savefig(buf,bbox_inches='tight', format='pdf')
                zf.writestr(f'{k}.pdf', buf.getvalue())
            else: raise
    zip_file.seek(0)
    b64 = base64.b64encode(zip_file.read()).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{label}.zip">Download {label}</a>'
    display(HTML(href))

def embed_multi_objs_folder(label,objs):
    import io
    import zipfile
    import pickle
    import os

    zip_file = io.BytesIO()
    with zipfile.ZipFile(zip_file, 'w') as zf:
        # iterate over the input dictionary
        for folder_name, obj_dict in objs.items():
            for obj_name, obj in obj_dict.items():
                if isinstance(obj, pd.DataFrame):
                    zf.writestr(f'{folder_name}\\{obj_name}.csv', obj.to_csv(index=False))
                elif isinstance(obj, pd.Series):
                    zf.writestr(f'{folder_name}\\{obj_name}.csv', obj.to_csv())
                elif isinstance(obj, plt.Figure):
                    buf = io.BytesIO()
                    obj.savefig(buf,bbox_inches='tight', format='pdf')
                    zf.writestr(f'{k}.pdf', buf.getvalue())
                else:
                    raise
    zip_file.seek(0)
    b64 = base64.b64encode(zip_file.read()).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{label}.zip">Download {label}</a>'
    display(HTML(href))
    
def make_toc(notebook_path):
    from .toci import Toci
    from IPython.display import Markdown
    toci = Toci()
    return Markdown(toci.execute(notebook_path))

def export_notebook(notebook_path, to = './figures/'):
    os.system(f"jupyter nbconvert {notebook_path}  --to html --no-input --output {os.path.join(to, notebook_path.replace('.ipynb','.html'))} --embed-images --log-level ERROR --no-prompt --template classic")
    
def saving_fig(
    f, project, file_name,
    to = '/share/home/biopharm/zhesi/results/',
    **kwargs
):
    write_dir = os.path.join(to, project, file_name)
    os.makedirs(os.path.join(to, project),mode=0o777, exist_ok=True)
    f.savefig(write_dir, bbox_inches='tight', **kwargs)
    
