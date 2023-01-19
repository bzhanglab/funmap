from os import path as osp
import re
import pandas as pd
from importlib import resources


def get_datafile_path(data_file):
    with resources.path('funmap.data', data_file) as f:
        data_file_path = f
    return data_file_path


def remove_version_id(value):
    if not pd.isna(value):
        return re.sub('\.\d+', '', value)
    else:
        return value


def gold_standard_edge_sets(gold_standard_file, id_type='ensembl_gene_id'):
    if id_type == 'ensembl_gene_id':
        gs_df = pd.read_csv(gold_standard_file, sep='\t')
        cols = gs_df.columns[0:2]
        gs_df.index = [tuple(sorted(x)) for x in zip(gs_df.pop(cols[0]), gs_df.pop(cols[1]))]
        gs_df = gs_df[~gs_df.index.duplicated(keep='first')]
        gs_pos_edges = set(gs_df.loc[gs_df.iloc[:, 0] == 1, :].index)
        gs_neg_edges = set(gs_df.loc[gs_df.iloc[:, 0] == 0, :].index)
    elif id_type == 'uniprot':
        gs_df = pd.read_csv(gold_standard_file)
        cols = gs_df.columns[0:2]
        gs_df.index = [tuple(sorted(x)) for x in zip(gs_df.pop(cols[0]), gs_df.pop(cols[1]))]
        gs_df = gs_df[~gs_df.index.duplicated(keep='first')]
        gs_pos_edges = set(gs_df.loc[gs_df.iloc[:, 0] == 'TP', :].index)
        gs_neg_edges = set(gs_df.loc[gs_df.iloc[:, 0] == 'FP', :].index)
    else:
        raise ValueError('id_type not supported')

    return gs_pos_edges, gs_neg_edges


def get_data_dict(data_config, min_sample_count):
    data_dict = {}
    data_root = data_config['data_root']
    mapping_file = get_datafile_path('knowledge_based_isoform_selection.txt')
    mapping = pd.read_csv(mapping_file, sep='\t')
    # gene ids are gene symbols
    for dt in data_config['data_files']:
        print(f'... {dt["name"]}')
        cur_feature = dt['name']
        cur_file = osp.join(data_root, dt['path'])
        cur_data = pd.read_csv(cur_file, sep='\t', index_col=0,
                                header=0)
        if cur_data.shape[1] < min_sample_count:
            print('...... not enough samples, skipped')
            continue
        cur_data = cur_data.T
        # exclude cohort with sample number < min_sample_count
        # remove noncoding genes first
        coding = mapping.loc[mapping['coding'] == 'coding', ['gene_name']]
        coding_genes = list(set(coding['gene_name'].to_list()))
        cur_data = cur_data[[c for c in cur_data.columns if c in coding_genes]]
        # duplicated columns, for now select the last column
        cur_data = cur_data.loc[:,~cur_data.columns.duplicated(keep='last')]
        data_dict[cur_feature] = cur_data

    return data_dict


# https://stackoverflow.com/a/312464/410069
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
