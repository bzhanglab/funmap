import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Any
import pandas as pd

# file hosted on figshare
urls = {
    'reactome_gold_standard': 'https://figshare.com/ndownloader/files/38647601',
    'funmap_blacklist': 'https://figshare.com/ndownloader/files/39033977',
    'mapping_file': 'https://figshare.com/ndownloader/files/39033971'
}

# https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """
    Calculate the hash of the given dictionary.

    Parameters
    ----------
    dictionary: Dict[str, Any]
        The input dictionary to be hashed.

    Returns
    -------
    str
        The 8-character hexadecimal hash of the dictionary.
    """
    dhash = hashlib.sha256()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()[:8]


def remove_version_id(value):
    """
    Remove ensembl version ID from value.

    Parameters
    ----------
    value : str
        The value, typically an ensembl ID, from which the version ID should be removed.

    Returns
    -------
    str
        The value with the version ID removed. If the value is NaN, it will be returned as is.
    """
    if not pd.isna(value):
        return re.sub('\.\d+', '', value)
    else:
        return value


def gold_standard_edge_sets(gold_standard_file, id_type='ensembl_gene_id'):
    """
    Extract positive and negative edges from a gold standard file.

    Parameters
    ----------
    gold_standard_file : str
        Path to the gold standard file.
    id_type : str, optional
        Type of IDs used in the gold standard file. Supported values are 'ensembl_gene_id' (default) and 'uniprot'.

    Returns
    -------
    gs_pos_edges : set
        Set of positive edges.
    gs_neg_edges : set
        Set of negative edges.

    Raises
    ------
    ValueError
        If `id_type` is not supported.

    Examples
    --------
    >>> gold_standard_file = 'path/to/gold_standard.tsv'
    >>> gs_pos_edges, gs_neg_edges = gold_standard_edge_sets(gold_standard_file, id_type='ensembl_gene_id')
    """
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


def get_data_dict(data_config, min_sample_count=15):
    """
    Returns a dictionary of data from the provided data configuration, filtered to only include genes that are
    coding and have at least `min_sample_count` samples.

    Parameters
    ----------
    data_config : dict
        A dictionary specifying the data file paths and names. It must contain the following keys:
        - 'data_root': the root path to the data files
        - 'data_files': a list of dictionaries, where each dictionary has keys 'name' and 'path'
    min_sample_count : int, optional
        Minimum number of samples required for a gene to be included in the returned dictionary. Default is 15.

    Returns
    -------
    data_dict : dict
        A dictionary where the keys are the names of the data files and the values are pandas DataFrames containing
        the data from the corresponding file.

    """
    data_dict = {}
    data_root = data_config['data_root']
    mapping = pd.read_csv(urls['mapping_file'], sep='\t')
    # gene ids are gene symbols
    for dt in data_config['data_files']:
        print(f"... {dt['name']}")
        cur_feature = dt['name']
        cur_file = Path(data_root) / dt['path']
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
    """
    Yield successive n-sized chunks from lst.

    Parameters
    ----------
    lst : list
        The input list to be split into chunks.
    n : int
        The size of each chunk.

    Yields
    ------
    list
        A chunk of size n from the input list lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
