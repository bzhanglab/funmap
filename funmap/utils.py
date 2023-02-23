import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import tarfile
from funmap.data_urls import misc_urls as urls


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


def get_data_dict(data_config, data_file, min_sample_count=15):
    """
    Returns a dictionary of data from the provided data configuration, filtered to only include genes that are
    coding and have at least `min_sample_count` samples.

    Parameters
    ----------
    data_config : dict
        A dictionary specifying the data file paths and names. It must contain the following keys:
        - 'data_root': the root path to the data files
        - 'data_files': a list of dictionaries, where each dictionary has keys 'name' and 'path'
    data_file : str/Path
        Path to the data file.
    min_sample_count : int, optional
        Minimum number of samples required for a gene to be included in the returned dictionary. Default is 15.

    Returns
    -------
    data_dict : dict
        A dictionary where the keys are the names of the data files and the values are pandas DataFrames containing
        the data from the corresponding file.

    """
    data_dict = {}
    mapping = pd.read_csv(urls['mapping_file'], sep='\t')
    # gene ids are gene symbols
    for dt in data_config['data_files']:
        print(f"... {dt['name']}")
        cur_feature = dt['name']
        cur_file = get_obj_from_tgz(data_file, dt['path'])
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


def get_node_edge(edge_list):
    """
    Calculate the number of nodes and edges, and the ratio of edges per node,
    and return the results in a dictionary format.

    Parameters
    ----------
    edge_list : pandas DataFrame
        The input DataFrame containing edge information.

    Returns
    -------
    dict
        A dictionary containing the number of nodes, the number of edges,
        the ratio of edges per node, a list of nodes, and the edge_list.

        The keys of the dictionary are:
        * n_node: int
            The number of nodes in the network.

        * n_edge: int
            The number of edges in the network.

        * edge_per_node: float
            The ratio of edges per node in the network.

        * nodes: list
            A list of nodes in the network.

        * edges: pandas DataFrame
            The edge_list input DataFrame.

    """
    # remove duplidated rows in edge_list
    edge_list = edge_list.drop_duplicates()
    n_edge = len(edge_list)
    nodes = set(edge_list.iloc[:,0].to_list()) | set(edge_list.iloc[:,1].to_list())
    return(dict(n_node = len(nodes), n_edge = n_edge,
        edge_per_node = n_edge / len(nodes),
        nodes = list(nodes),
        edges = edge_list))


def get_node_edge_overlap(network_info):
    """
    Computes the node and edge overlap between networks.

    Parameters
    ----------
    network_info : pandas DataFrame
        A DataFrame with information about the networks.

    Returns
    -------
    overlap : dict
        A dictionary with the node and edge overlap between networks.
    """
    networks = pd.DataFrame(columns = ['name', 'type', 'n_node',
                                    'n_edge', 'edge_per_node',
                                    'nodes', 'edges'])

    for _, row in network_info.iterrows():
        network_name = row['name']
        network_type = row['type']
        network_el = row['el']
        res = get_node_edge(network_el)
        cur_df = pd.DataFrame({'name': [network_name],
                                'type': [network_type],
                                'n_node': [int(res['n_node'])],
                                'n_edge': [int(res['n_edge'])],
                                'edge_per_node': [res['edge_per_node']],
                                'nodes': [res['nodes']],
                                'edges': [res['edges']]})
        networks = pd.concat([networks, cur_df], ignore_index=True)
    # overlap of nodes and edges
    overlap = {}

    # node overlap
    target = 'FunMap'
    cur_res = {}
    target_node_set = set(networks.loc[networks['name'] == target,
                                    'nodes'].tolist()[0])
    target_size = len(target_node_set)
    for _, row in networks.iterrows():
        if row['name'] == target:
            continue
        cur_node_set = set(row['nodes'])
        cur_size = len(cur_node_set)
        overlap_size = len(target_node_set & cur_node_set)
        cur_res[row['name']] = tuple([
                target_size - overlap_size,
                cur_size - overlap_size,
                overlap_size])

    overlap['node'] = cur_res

    # edge overlap
    cur_res = {}
    target_edge_df = networks.loc[networks['name'] == target, 'edges'].tolist()[0]
    target_edge_set = set(tuple(sorted(x)) for x in zip(target_edge_df.pop(0),
                                                    target_edge_df.pop(1)))
    target_size = len(target_edge_set)

    for _, row in networks.iterrows():
        if row['name'] == target:
            continue
        edge_df = row['edges']
        edges = [tuple(sorted(x)) for x in zip(edge_df.pop(0), edge_df.pop(1))]
        cur_edge_set = set(edges)
        cur_size = len(cur_edge_set)
        overlap_size = len(target_edge_set & cur_edge_set)
        cur_res[row['name']] = tuple([
            target_size - overlap_size,
            cur_size - overlap_size,
            overlap_size])

    overlap['edge'] = cur_res
    return overlap


# write a function that read a specific file from a tar gz file and return
# the file object
def get_obj_from_tgz(tar_file, file_name):
    """
    Read a specific file from a tar gz file and return the file object.

    Parameters
    ----------
    tar_file : str
        The path to the tar gz file.

    file_name : str
        The name of the file to be read.

    Returns
    -------
    f
        The file object of the file to be read.
    """

    tar = tarfile.open(tar_file, "r:gz")
    for member in tar.getmembers():
        if member.name.endswith(file_name):
            f = tar.extractfile(member)
            if f is not None:
                return f
