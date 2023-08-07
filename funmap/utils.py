import hashlib
import csv
import yaml
import os
import tarfile
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import shutil
from funmap.data_urls import misc_urls as urls
from funmap.logger import setup_logger

log = setup_logger(__name__)

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
    if 'filter_noncoding_genes' in data_config and data_config['filter_noncoding_genes']:
        mapping = pd.read_csv(urls['mapping_file'], sep='\t')
    # extract the data file from the tar.gz file
    tmp_dir = 'tmp_data'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    os.system(f'tar -xzf {data_file} --strip-components=1 -C {tmp_dir}')
    # gene ids are gene symbols
    for dt in data_config['data_files']:
        log.info(f"processing ... {dt['name']}")
        cur_feature = dt['name']
        # cur_file = get_obj_from_tgz(data_file, dt['path'])
        # extract the data file from the tar.gz file
        # cur_data= get_obj_from_tgz(data_file, dt['path'])
        cur_data = pd.read_csv(os.path.join(tmp_dir, dt['path']), sep='\t', index_col=0,
                                header=0)
        if cur_data.shape[1] < min_sample_count:
            log.info(f"... {dt['name']} ... not enough samples, skipped")
            continue
        cur_data = cur_data.T
        # exclude cohort with sample number < min_sample_count
        # remove noncoding genes first
        if data_config['filter_noncoding_genes']:
            coding = mapping.loc[mapping['coding'] == 'coding', ['gene_name']]
            coding_genes = list(set(coding['gene_name'].to_list()))
            cur_data = cur_data[[c for c in cur_data.columns if c in coding_genes]]
        # duplicated columns, for now select the last column
        cur_data = cur_data.loc[:,~cur_data.columns.duplicated(keep='last')]
        data_dict[cur_feature] = cur_data

    shutil.rmtree(tmp_dir)

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

def cleanup_experiment(config_file, data_file):
    cfg = get_config(config_file, data_file)
    results_dir = Path(cfg['results_dir'])
    shutil.rmtree(results_dir, ignore_errors=True)

def setup_experiment(config_file, data_file):
    cfg = get_config(config_file, data_file)
    results_dir = Path(cfg['results_dir'])
    # create folders
    folder_dict = cfg['subdirs']
    folders = [results_dir / Path(folder_dict[folder_name]) for folder_name in folder_dict]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    # save configuration to results folder
    with open(str( results_dir / 'config.yml'), 'w') as fh:
        yaml.dump(cfg, fh, sort_keys=False)

    return cfg

def get_config(cfg_file: Path, data_file: Path):
    cfg = {
        'filter_after_prediction': True,
        'filter_criterion': 'max',
        'filter_threshold': 0.95,
        'filter_blacklist': True,
        'n_jobs': os.cpu_count(),
        'n_chunks': 4,
        'start_edge_num': 10000,
        'max_num_edges': 250000,
        'step_size': 100,
        'lr_cutoff': 50,
        'results_dir': 'results',
        # the following directories are relative to the results_dir
        'subdirs': {
            'saved_model_dir': 'saved_models',
            'saved_data_dir': 'saved_data',
            'saved_predictions_dir': 'saved_predictions',
            'figure_dir': 'figures',
            'network_dir': 'networks',
        },
        'seed': 42,
        'cor_type': 'pearson',
        'feature_type': 'cc',
        'split_by': 'edge',
        'test_size': 0.5,
        'ml_type': 'xgboost',
        'filter_before_prediction': True,
        'min_feature_count': 1,
        'min_sample_count': 20,
        'filter_noncoding_genes': False
    }

    with open(cfg_file, 'r') as fh:
        cfg_dict = yaml.load(fh, Loader=yaml.FullLoader)

    if 'seed' in cfg_dict:
        cfg['seed'] = cfg_dict['seed']

    if 'cor_type' in cfg_dict:
        cfg['cor_type'] = cfg_dict['cor_type']

    if 'min_sample_count' in cfg_dict:
        cfg['min_sample_count'] = cfg_dict['min_sample_count']

    if 'n_jobs' in cfg_dict:
        cfg['n_jobs'] = cfg_dict['n_jobs']

    if 'n_chunks' in cfg_dict:
        cfg['n_chunks'] = cfg_dict['n_chunks']

    if 'start_edge_num' in cfg_dict:
        cfg['start_edge_num'] = cfg_dict['start_edge_num']

    if 'max_num_edges' in cfg_dict:
        cfg['max_num_edges'] = cfg_dict['max_num_edges']

    if 'step_size' in cfg_dict:
        cfg['step_size'] = cfg_dict['step_size']

    if 'lr_cutoff' in cfg_dict:
        cfg['lr_cutoff'] = cfg_dict['lr_cutoff']

    if 'name' in cfg_dict:
        cfg['name'] = cfg_dict['name']
    else:
        raise ValueError('name not specified in config file')

    if 'filter_noncoding_gene' in cfg_dict:
        cfg['filter_noncoding_genes'] = cfg_dict['filter_noncoding_genes']

    if 'results_dir' in cfg_dict:
        cfg['results_dir'] = cfg_dict['results_dir'] + '/' + cfg_dict['name']

    if 'data_files' not in cfg_dict:
        raise ValueError('data_files not specified in config file')

    # Check all files listed under data_files are also in the tar.gz file
    data_files = cfg_dict['data_files']
    # List all the files in the tar.gz file
    with tarfile.open(data_file, "r:gz") as tar:
        tar_files = {Path(file).name for file in tar.getnames()}

    # Check if all files in data_files are in tar_files
    if not all(file['path'] in tar_files for file in data_files):
        print('Files listed under data_files are not in the tar.gz file!')
        raise ValueError('Files listed under data_files are not in the tar.gz file!')

    cfg['data_files'] = cfg_dict['data_files']

    if 'rp_pairs' in cfg_dict:
        cfg['rp_pairs'] = cfg_dict['rp_pairs']

    return cfg


def check_gold_standard_file(file_path, min_count=10000):
    """
    min_threshold : int
        The minimum threshold for the lesser of '0' and '1' counts in the 'Class' column.

    """
    try:
        with open(file_path, 'r', newline='') as tsv_file:
            dialect = csv.Sniffer().sniff(tsv_file.read(1024))
            if dialect.delimiter != '\t':
                print("Error: Incorrect TSV format. TSV files should be tab-separated.")
                return False
    except csv.Error:
        print("Error: Unable to read TSV file.")
        return False

    # Check data format and Class values
    class_values = []
    with open(file_path, 'r', newline='') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        next(reader)  # Skip header
        for row_num, row in enumerate(reader, start=2):  # Add row_num for better error reporting
            if len(row) != 3:  # Assuming each row should have 3 columns
                log.error(f'Invalid row format in row {row_num}. Each row should have 3 columns.')
                return False
            class_value = row[2].strip()
            if not class_value.isdigit() or int(class_value) not in (0, 1):
                log.error(f'Invalid "Class" value in row {row_num}. Must be 0 or 1.')
                return False
            class_values.append(int(class_value))

    # Check Class value counts and ratio
    count_0 = class_values.count(0)
    count_1 = class_values.count(1)
    lesser_count = min(count_0, count_1)

    if lesser_count < min_count:
        log.error(f"The lesser of 0 and 1 occurrences ({lesser_count}) does not meet the threshold. "
            f"Expected at least {min_count}.")
        return False

    return True


def check_extra_feature_file(file_path, missing_value='NA'):
    """
    Notes
    -----
    This function checks the following criteria for the measurement TSV file:
    - The file must have a header row.
    - There must be at least 3 columns.
    - The first two columns are gene/protein IDs.
    - The data type for each additional column (excluding the first two columns) must be consistent
    and can be either integer, float, or the specified missing_value.

    If any of the checks fail, the function will print informative error messages and return False.

    The TSV file should be tab-separated.
    """
    try:
        with open(file_path, 'r', newline='') as tsv_file:
            dialect = csv.Sniffer().sniff(tsv_file.read(1024))
            if dialect.delimiter != '\t':
                log.error("Incorrect TSV format. TSV files should be tab-separated.")
                return False
    except csv.Error:
        log.error("Unable to read TSV file.")
        return False

    # Check header and number of columns
    with open(file_path, 'r', newline='') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        header = next(reader, None)
        if header is None:
            log.error("The TSV file must have a header row.")
            return False

        num_columns = len(header)
        if num_columns < 3:
            log.error("The TSV file must have at least 3 columns.")
            return False

    # Check data type consistency for additional columns
    with open(file_path, 'r', newline='') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        column_data_types = {}
        for row_num, row in enumerate(reader, start=2):  # Add row_num for better error reporting
            for column_name, value in row.items():
                if column_name not in header[:2]:  # Skip the first two columns (Protein_1 and Protein_2)
                    if value == missing_value:
                        continue
                    try:
                        float_value = float(value)
                        if float_value.is_integer():
                            value = int(float_value)
                    except ValueError:
                        log.error("Invalid data type in row %d, column '%s'. "
                                    "The value '%s' should be either an integer, a float, or '%s' (missing value).",
                                    row_num, column_name, value, missing_value)
                        return False
                    # Store the data type of each column (float or integer)
                    data_type = float if '.' in value else int
                    if column_name not in column_data_types:
                        column_data_types[column_name] = data_type
                    elif column_data_types[column_name] != data_type:
                        log.error("Inconsistent data type in column '%s'. "
                                    "Expected a consistent data type (integer, float, or '%s') for all rows.",
                                    column_name, missing_value)
                        return False

    return True
