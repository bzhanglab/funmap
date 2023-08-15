import os
import glob
import re
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.utils import resample
import itertools
import warnings
from typing import List
import gzip, pickle
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from funmap.utils import get_data_dict
from funmap.data_urls import network_info, misc_urls as urls
from funmap.logger import setup_logger

log = setup_logger(__name__)

def get_valid_gs_data(gs_path: str, valid_gene_list: List[str]):
    """
    Get valid gene-gene pairs by removing non-valid genes and removing duplicate edges.

    Parameters
    ----------
    gs_path : str
        The path of the gene-gene pair file
    valid_gene_list : List[str]
        List of valid genes.

    Returns
    -------
    gs_edge_df : pd.DataFrame
        Dataframe containing valid gene-gene pairs
    """
    log.info(f'Loading gold standard feature file "{gs_path}" ...')
    gs_edge_df = pd.read_csv(gs_path, sep='\t')
    log.info('Done loading gold standard feature file')
    gs_edge_df = gs_edge_df.rename(columns={gs_edge_df.columns[0]: 'P1',
                                            gs_edge_df.columns[1]: 'P2'})
    gs_edge_df = gs_edge_df[gs_edge_df['P1'].isin(valid_gene_list) &
                            gs_edge_df['P2'].isin(valid_gene_list) &
                            (gs_edge_df['P1'] != gs_edge_df['P2'])]
    m = ~pd.DataFrame(np.sort(gs_edge_df[['P1','P2']], axis=1)).duplicated()
    gs_edge_df = gs_edge_df[list(m)]
    gs_edge_df.reset_index(drop=True, inplace=True)
    # rename the last column name to 'label'
    gs_edge_df.rename(columns={gs_edge_df.columns[-1]: 'label'}, inplace=True)

    return gs_edge_df


def pairwise_mutual_rank(pcc_matrix):
    """
    Calculate the pairwise mutual rank matrix based on the given Pearson correlation coefficient matrix.

    Parameters:
    -----------
    pcc_matrix : numpy.ndarray
        The Pearson correlation coefficient matrix. It should be a square matrix where
        pcc_matrix[i, j] represents the correlation coefficient between variables i and j.

    Returns:
    --------
    numpy.ndarray
        A matrix containing the pairwise mutual ranks between variables based on the
        provided Pearson correlation coefficient matrix. The matrix has the same shape
        as the input pcc_matrix.

    Mutual Rank Calculation:
    ------------------------
    The mutual rank between two variables A and B, based on their Pearson correlation coefficients,
    is a measure of their relative rankings within their respective groups of correlated variables.
    The formula for calculating the mutual rank is given by:

    mr_{AB} = sqrt((r_{AB} / n_B) * (r_{BA} / n_A))

    Where:
    - mr_{AB} is the mutual rank between variables A and B.
    - r_{AB} is the rank of the correlation coefficient between A and B among all other correlation
        coefficients involving A (excluding NaN values).
    - n_B is the number of valid (non-NaN) correlation coefficients involving variable B.
    - r_{BA} is the rank of the correlation coefficient between B and A among all other correlation
        coefficients involving B (excluding NaN values).
    - n_A is the number of valid (non-NaN) correlation coefficients involving variable A.

    Steps:
    - For each variable pair (A, B):
        - Calculate the rank of the correlation coefficient between A and B among all other correlation
        coefficients involving A. This rank is denoted as r_{AB}.
        - Calculate the rank of the correlation coefficient between B and A among all other correlation
        coefficients involving B. This rank is denoted as r_{BA}.
    - For each variable pair (A, B):
        - Determine the number of valid (non-NaN) correlation coefficients involving variable B, denoted as n_B.
        - Determine the number of valid (non-NaN) correlation coefficients involving variable A, denoted as n_A.
    - For each variable pair (A, B):
        - Compute the mutual rank mr_{AB} using the formula mentioned earlier.
    - Populate the mutual rank matrix:
        - Create a new matrix with the same shape as the input correlation coefficient matrix,
        initialized with NaN values.
        - For each valid variable pair (A, B), assign the corresponding mutual rank mr_{AB}
        to the matrix at the appropriate indices.

    The resulting matrix contains the mutual ranks between all pairs of variables based on their
    Pearson correlation coefficients. Higher mutual rank values indicate stronger and more consistent
    correlations between variables.
    """
    valid_a = ~np.isnan(pcc_matrix)
    valid_b = valid_a.T

    rank_ab = np.argsort(pcc_matrix, axis=1).argsort(axis=1, kind='mergesort') + 1  # Start ranks from 1
    rank_ba = np.argsort(pcc_matrix, axis=0).argsort(axis=0, kind='mergesort') + 1  # Start ranks from 1

    n_a = np.sum(valid_a, axis=1)
    n_b = np.sum(valid_b, axis=0)

    valid_indices_a, valid_indices_b = np.where(valid_a)

    mr_values = np.sqrt((rank_ab[valid_indices_a, valid_indices_b] / n_b[valid_indices_b]) *
                        (rank_ba[valid_indices_a, valid_indices_b] / n_a[valid_indices_a]))

    mr_matrix = np.full_like(pcc_matrix, np.nan)
    mr_matrix[valid_indices_a, valid_indices_b] = mr_values

    return mr_matrix


def compute_features(cfg, feature_type, min_sample_count, output_dir):
    """Compute the pearson correlation coefficient for each edge in the list of edges and for each
    """
    data_dict, all_valid_ids = get_data_dict(cfg, min_sample_count)
    cc_dict = {}
    for i in data_dict:
        cc_file = os.path.join(output_dir, f'cc_{i}.h5')
        cc_dict[i] = cc_file

    mr_dict = {}
    for i in data_dict:
        mr_file = os.path.join(output_dir, f'mr_{i}.h5')
        mr_dict[i] = mr_file

    all_cc_exist = all(os.path.exists(file_path) for file_path in cc_dict.values())
    if all_cc_exist:
        log.info("All cc files exist. Skipping feature computation.")
        if feature_type == 'cc':
            return cc_dict, mr_dict, all_valid_ids

    if feature_type == 'mr':
        all_mr_exist = all(os.path.exists(file_path) for file_path in mr_dict.values())
        if all_cc_exist and all_mr_exist:
            log.info("All mr files exist. Skipping feature computation.")
            return cc_dict, mr_dict, all_valid_ids

    log.debug(f"Computing {feature_type} features")
    for i in data_dict:
        cc_file = cc_dict[i]
        if os.path.exists(cc_file):
            continue
        log.info(f"Computing pearson correlation coefficient matrix for {i}")
        x = data_dict[i].values.astype(np.float32)
        x = x.T
        xave = np.nanmean(x, axis=1)
        xstd = np.nanstd(x, axis=1)
        ztrans = x.T - xave
        # it is possible that xstd is 0, e.g. all except one value is nan
        # ignore the warning
        with np.errstate(invalid='ignore'):
            ztrans /= xstd
        z = ztrans.T
        z = np.ma.array(z, mask=np.isnan(z))
        arr = np.ma.dot(z, z.T)
        valid_pairs_matrix = np.sum(~np.logical_or(z.mask[:, np.newaxis], z.mask), axis=2)
        arr /= valid_pairs_matrix
        arr = np.array(arr, dtype=np.float32)
        arr[valid_pairs_matrix < min_sample_count] = np.nan
        upper_indices = np.triu_indices(arr.shape[0])
        with h5py.File(cc_dict[i], 'w') as hf:
            # only store the upper triangle part
            hf.create_dataset('cc',  data=arr[upper_indices])
            hf.create_dataset('ids', data=data_dict[i].columns.values.astype('S'))
        cc_dict[i] = cc_file

        # compute pairwise mutual rank features
        if feature_type == 'mr':
            log.info(f"Computing mutual rank matrix for {i}")
            arr_mr = pairwise_mutual_rank(arr)
            upper_indices = np.triu_indices(arr_mr.shape[0])
            with h5py.File(mr_dict[i], 'w') as hf:
                # only store the upper triangle part
                hf.create_dataset('mr',  data=arr_mr[upper_indices])
                hf.create_dataset('ids', data=data_dict[i].columns.values.astype('S'))
            mr_dict[i] = mr_file

    return cc_dict, mr_dict, all_valid_ids


def balance_classes(df, random_state=42):
    class_column = df.columns[-1]  # Assuming class column is the last column

    class_values = df[class_column].unique()
    if len(class_values) != 2:
        raise ValueError("The class column should have exactly 2 unique values.")

    class_0 = df[df[class_column] == class_values[0]]
    class_1 = df[df[class_column] == class_values[1]]
    minority_class = class_0 if len(class_0) < len(class_1) else class_1
    majority_class = class_1 if minority_class is class_0 else class_0

    majority_class_undersampled = resample(majority_class, replace=False,
                            n_samples=len(minority_class), random_state=random_state)

    balanced_df = pd.concat([minority_class, majority_class_undersampled])
    # Shuffle the rows in the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

def assemble_feature_df(h5_file_mapping, gs_df, dataset='cc', extra_feature_file=None):
    gs_df.reset_index(drop=True, inplace=True)
    # Initialize feature_df with columns for HDF5 file keys and 'label'
    file_keys = list(h5_file_mapping.keys())
    feature_df = pd.DataFrame(columns=file_keys + ['label'])

    # Iterate over HDF5 files and load feature values
    for key, file_path in h5_file_mapping.items():
        with h5py.File(file_path, 'r') as h5_file:
            gene_ids = h5_file['ids'][:]
            gene_to_index = {gene.astype(str): idx for idx, gene in enumerate(gene_ids)}

            # Get gene indices for P1 and P2
            p1_indices = np.array([gene_to_index.get(gene, -1) for gene in gs_df.iloc[:, 0]])
            p2_indices = np.array([gene_to_index.get(gene, -1) for gene in gs_df.iloc[:, 1]])

            f_dataset = h5_file[dataset]
            f_values = np.empty(len(gs_df), dtype=float)
            valid_indices = (p1_indices != -1) & (p2_indices != -1)

            linear_indices_func = lambda row_indices, col_indices, n: np.array(col_indices) - np.array(row_indices) + (2*n - np.array(row_indices) + 1) * np.array(row_indices) // 2
            linear_indices = linear_indices_func(p1_indices[valid_indices], p2_indices[valid_indices], len(gene_ids))

            f_values[valid_indices] = f_dataset[:][linear_indices]
            f_values[~valid_indices] = np.nan

            # Add feature values to the feature_df
            feature_df[key] = f_values

    # if the last column is 'label', assign it to feature_df
    if gs_df.columns[-1] == 'label':
        feature_df['label'] = gs_df[gs_df.columns[-1]]
    else:
        # delete the 'label' column from feature_df
        del feature_df['label']

    # TODO: Add extra features if provided
    if extra_feature_file is not None:
        pass

    return feature_df

def extract_features(df, feature_type, cc_dict, extra_feature_file, mr_dict):
    if feature_type == 'mr':
        if not mr_dict:
            raise ValueError('mr dict is empty')

    feature_dict = cc_dict if feature_type == 'cc' else mr_dict
    feature_df = assemble_feature_df(feature_dict, df, feature_type, extra_feature_file)
    return feature_df


def get_ppi_feature():
    """
    Returns a dictionary of protein-protein interaction (PPI) features.

    The PPI features are extracted from data in the "network_info" dictionary and are specified by the
    "feature_names" list. The URLs of the relevant data are extracted from "network_info" and read
    using the Pandas library. The resulting PPI data is stored in the "ppi_features" dictionary and
    returned by the function.

    Returns:
    ppi_features: dict
        A dictionary with PPI features, where the keys are the feature names and the values are lists of tuples
        representing the protein interactions.
    """
    feature_names = ['BioGRID', 'BioPlex', 'HI-union']
    urls = [network_info['url'][i] for i in range(len(network_info['name']))
                if network_info['name'][i] in feature_names]

    ppi_features = {}
    # use pandas to read the file
    for (i, url) in enumerate(urls):
        data = pd.read_csv(url, sep='\t', header=None)
        data = data.apply(lambda x: tuple(sorted(x)), axis=1)
        ppi_name = f'{feature_names[i]}_PPI'
        ppi_features[ppi_name] = data.tolist()

    from IPython import embed; embed()
    import sys; sys.exit(0)

    return ppi_features


def train_ml_model(data_df, ml_type, seed, n_jobs):
    """
    Train a machine learning model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataframe containing the input features and target variable.
    ml_type : str
        Type of machine learning model to use. Currently only 'xgboost' is supported.
    seed : int
        Seed for the random number generator for reproducibility.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    models: dict
        A dictionary of trained models, where the keys are the target
        variables and the values are the

    """
    assert ml_type == 'xgboost', 'ML model must be xgboost'
    log.info(f'Training model with {n_jobs} jobs ...')
    models = train_model(data_df.iloc[:, :-1], data_df.iloc[:, -1], seed, n_jobs)
    log.info('Training model ... done')

    return models


def train_model(X, y, seed, n_jobs):
    """
    Train a XGBoost model using the input feature matrix X and target vector y.
    The model is trained using GridSearchCV with a specified set of parameters,
    and a 5-fold stratified cross validation.

    Parameters
    -----------
    X : pd.DataFrame
        The feature matrix
    y : pd.Series
        The target vector
    seed : int
        The random seed used for reproducibility
    n_jobs : int
        The number of CPU cores used for parallel computation

    Returns
    --------
    models: dict
        A dictionary of trained models, where the keys are the feature types
        and the values are the trained models.
    """
    model_params = {
        'n_estimators': [50, 150, 250, 300],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    }

    # use only mutual rank
    xgb_model = xgb.XGBClassifier(random_state=seed,
                            eval_metric='logloss', n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    clf = GridSearchCV(xgb_model, model_params, scoring='roc_auc', cv=cv,
                    n_jobs=1, verbose=2)
    model = clf.fit(X, y)

    return model


def validation_llr(all_feature_df, predicted_all_pairs,
                filter_after_prediction, filter_criterion, filter_threshold,
                filter_blacklist, blacklist_file: str, max_num_edges, step_size,
                gs_test_pos_set, gs_test_neg_set, output_dir: Path,
                network_out_dir: Path):
    """
    Compute Log Likelihood Ratio (LLR) for a given set of edges and a given set of gold
    standard positive and negative edges.
    The function performs filtering and sorting on the input edges before computing LLR.

    Parameters
    ----------
    all_feature_df (pd.DataFrame): Dataframe containing all features
    predicted_all_pair (pd.DataFrame): Dataframe containing predicted values of edges
    filter_after_prediction (bool): whether to filter edges after prediction
    filter_criterion (str): criterion to filter edges
    filter_threshold (float): threshold value to filter edges
    filter_blacklist (bool): whether to filter edges incident on genes in blacklist
    blacklist_file (str): url to blacklist file
    max_num_edges (int): maximum number of edges to compute LLR for
    step_size (int): step size for iterating over edges
    gs_test_pos_set (set): set of gold standard positive edges
    gs_test_neg_set (set): set of gold standard negative edges
    output_dir (Path): directory to save LLR results
    network_out_dir (Path): directory to save selected edges

    Returns
    -------
    llr_res_dict (pd.DataFrame): Dataframe containing LLR results for all selected edges
    edge_list_file_out (Path): path to save selected edges
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    ret = {}

    for ft in predicted_all_pairs:
        # final results
        llr_res_file = output_dir / f'llr_results_{ft}_{max_num_edges}.tsv'

        print(f'Calculating llr_res_dict ({ft})...')
        llr_res_dict = {}
        cur_col_name = 'prediction'
        cur_results = predicted_all_pairs[ft][[cur_col_name]].copy()
        cur_results.sort_values(by=cur_col_name, ascending=False,
                                inplace=True)
        # for this filter criterion, only based on 'MR'
        if filter_after_prediction:
            if filter_criterion != 'max':
                raise ValueError('Filter criterion must be "max" for MR')
            # filter edges with MR < filter_threshold
            if ft == 'ex':
                regex_str = feature_type_to_regex(ft)
                all_feature_df_sel = all_feature_df.filter(regex=regex_str)
                all_feature_df_sel = all_feature_df_sel.drop(all_feature_df_sel[all_feature_df_sel.max(axis=1)
                                    < filter_threshold].index)
                cur_results = cur_results[cur_results.index.isin(all_feature_df_sel.index)]
            elif ft == 'ei':
                # the filter is still based on 'ex', but we need to add back
                # _PPI columns back after the filtering
                regex_str = feature_type_to_regex('ex')
                all_feature_df_sel = all_feature_df.filter(regex=regex_str)
                all_feature_df_sel = all_feature_df_sel.drop(all_feature_df_sel[all_feature_df_sel.max(axis=1)
                                    < filter_threshold].index)

                regex_str2 = feature_type_to_regex('ei')
                all_feature_df_sel_2 = all_feature_df.filter(regex=regex_str2)
                # only keep indices that are in all_feature_df_sel
                all_feature_df_sel_2 = all_feature_df_sel_2[all_feature_df_sel_2.index.isin(all_feature_df_sel.index)]
                cur_results = cur_results[cur_results.index.isin(all_feature_df_sel_2.index)]
            else:
                raise ValueError(f'Filtering not supported for {ft}')

        # remove any edge that is incident on any genes in the black list
        if filter_blacklist:
            bl_genes = pd.read_csv(blacklist_file, sep='\t', header=None)
            bl_genes = set(bl_genes[0].to_list())
            cur_results = cur_results.reset_index()
            cur_results[['e1', 'e2']] = pd.DataFrame(cur_results['index'].tolist(),
                                        index=cur_results.index)
            cur_results = cur_results[~(cur_results['e1'].isin(bl_genes)
                                    | cur_results['e2'].isin(bl_genes))]
            cur_results.drop(columns=['e1', 'e2'], inplace=True)
            cur_results = cur_results.set_index('index')

        cnt_notna = np.count_nonzero(~np.isnan(cur_results.values))
        print(f'total number of pairs with valid prediction: {cnt_notna}')
        result_dict = defaultdict(list)
        assert cnt_notna > max_num_edges, f'not enough valid edges after filtering, need {max_num_edges}, actual {cnt_notna}'
        # llr_res_dict only save maximum of max_steps data points for downstream
        # analysis / plotting
        for k in tqdm(range(step_size, max_num_edges+step_size, step_size)):
            selected_edges = set(cur_results.iloc[:k, :].index)
            all_nodes = set(itertools.chain.from_iterable(selected_edges))
            # https://stackoverflow.com/a/7590970/410069
            common_pos_edges = selected_edges & gs_test_pos_set
            common_neg_edges = selected_edges & gs_test_neg_set
            try:
                lr = len(common_pos_edges) / len(common_neg_edges) / (len(gs_test_pos_set) / len(gs_test_neg_set))
            except ZeroDivisionError:
                lr = 0
            llr = np.log(lr) if lr > 0 else np.nan
            n_node = len(all_nodes)
            result_dict['k'].append(k)
            result_dict['n'].append(n_node)
            result_dict['llr'].append(llr)

        llr_res_dict = pd.DataFrame(result_dict)
        llr_res_dict.to_csv(llr_res_file, sep='\t', index=False)

        # write edge list to file, also include the prediction score
        print(f'saving edges to file ...')
        edge_list_file = network_out_dir / f'network_{ft}_{max_num_edges}.tsv'
        cur_results = cur_results.reset_index()
        cur_results[['e1', 'e2']] = pd.DataFrame(cur_results['index'].tolist(),
                                    index=cur_results.index)
        cur_results.drop(columns=['index'], axis=1, inplace=True)
        cur_results = cur_results.reindex(columns=['e1', 'e2', cur_col_name])
        selected_edges = cur_results.iloc[:max_num_edges, :]
        selected_edges.to_csv(edge_list_file, sep='\t', index=False, header=False)
        print(f'Calculating llr_res_dict ({ft})... done')

        ret[ft] = {
            'llr_res_path': llr_res_file,
            'edge_list_path': edge_list_file
        }

    return ret


# set the final funmap edge list file based on the cutoff
def get_funmap(validation_res, config, output_dir):
    """Set the final funmap edge list file based on the likelihood ratio cutoff.
    we will use results from "ei" to generate the final edge list.

    Parameters
    ----------
    validation_res : dict
        A dictionary containing the validation results. It should have a key 'ei' which
        should be a dictionary containing the paths to the likelihood ratio test results
        and the edge list file.
    run_config : dict
        A dictionary containing the configuration for running the function. It should have
        a key 'lr_cutoff' which specifies the likelihood ratio cutoff value.
    output_dir : Path
        A Path object specifying the output directory where the final funmap edge list file
        will be saved.

    Returns
    -------
    None
        This function does not return anything, but it saves the final funmap edge list file
        to the output directory specified by `output_dir`.

    Raises
    ------
    UserWarning
        If the largest llr value is smaller than the cutoff, no funmap will be generated.
    """
    llr_res_path = validation_res['ei']['llr_res_path']
    edge_list_path = validation_res['ei']['edge_list_path']
    llr_res = pd.read_csv(llr_res_path, sep='\t')
    llr_res = llr_res.sort_values(by=['llr'], ascending=False)
    cutoff = config['lr_cutoff']

    if llr_res['llr'].iloc[0] < np.log(cutoff):
        warnings.warn('The largest llr value is smaller than the cutoff, no funmap will be generated.')
        return
    llr_res = llr_res[llr_res['llr'] >= np.log(cutoff)]
    funmap = llr_res.iloc[-1]
    n_edge = int(funmap['k'])
    funmap_el = pd.read_csv(edge_list_path, sep='\t', header=None)
    funmap_el = funmap_el.iloc[:n_edge, :2]
    funmap_el.to_csv(output_dir / 'funmap.tsv', sep='\t', header=False, index=False)


def prepare_gs_data(**kwargs):
    cc_dict = kwargs['cc_dict']
    mr_dict = kwargs['mr_dict']
    gs_file = kwargs['gs_file']
    feature_type = kwargs['feature_type']
    extra_feature_file = kwargs['extra_feature_file']
    valid_id_list = kwargs['valid_id_list']
    test_size = kwargs['test_size']
    seed = kwargs['seed']

    if gs_file is None:
        gs_file = urls['reactome_gold_standard']

    # TODO:  use user provided gs_file if it is not None

    log.info('Preparing gold standard data')
    gs_df = get_valid_gs_data(gs_file, valid_id_list)
    gs_df_balanced = balance_classes(gs_df, random_state=seed)
    del gs_df
    gs_train, gs_test = train_test_split(gs_df_balanced,
            test_size=test_size, random_state=seed,
            stratify=gs_df_balanced.iloc[:, -1])

    gs_train_df = extract_features(gs_train, feature_type, cc_dict,
                                    extra_feature_file, mr_dict)

    gs_test_df =  extract_features(gs_test, feature_type, cc_dict,
                                    extra_feature_file, mr_dict)
    # store both the ids with gs_test_df for later use
    # add the first two column of gs_test to gs_test_df at the beginning
    gs_test_df = pd.concat([gs_test.iloc[:, :2], gs_test_df], axis=1)

    # cols = gs_test.columns[0:2]
    # gs_test.index = [tuple(sorted(x)) for x in
    #                     zip(gs_test.pop(cols[0]),
    #                         gs_test.pop(cols[1]))]
    # label_col = gs_test.columns[-1]
    # gs_test_pos_df = gs_test.loc[gs_test[label_col] == 1, label_col]
    # gs_test_neg_df = gs_test.loc[gs_test[label_col] == 0, label_col]
    # return gs_train_df, gs_test_pos_df, gs_test_neg_df
    log.info('Preparing gs data ... done')
    return gs_train_df, gs_test_df


def dataset_llr(feature_df, gs_test_pos_set, gs_test_neg_set,
                start_edge_num, step_size, max_num_edge,
                output_file='llr_dataset.tsv'):
    """Calculate the Log-Likelihood Ratio (LLR) for a set of MR features.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        DataFrame containing features for calculation.
    gs_test_pos_set : set
        A set of positive test edges.
    gs_test_neg_set : set
        A set of negative test edges.
    start_edge_num : int
        Start number of edges.
    step_size : int
        Step size.
    max_num_edge : int
        Maximum number of edges.
    output_file : str, optional
        Output file name, by default 'llr_dataset.tsv'.

    Returns
    -------
    llr_ds: pandas.DataFrame

    Notes
    -----
    - feature_df should contain features ending with "_MR".
    - Features ending with "_CC" will be removed from calculation.
    - start_edge_num should be smaller than max_num_edge.
    - max_num_edge should be smaller than the number of non-NA values.
    """
    mr_df = feature_df.filter(regex='_MR$', axis=1)
    result_dict = defaultdict(list)
    for col in mr_df:
        dataset_name = re.sub('_MR$', '', col)
        print(f'... processing {dataset_name}')
        cur_results = mr_df[[col]].copy()
        cur_results.sort_values(by=col, ascending=False, inplace=True)
        cnt_notna = np.count_nonzero(~np.isnan(cur_results[col].values))
        assert start_edge_num < max_num_edge, 'start_edge_num should be smaller than max_num_edge'
        print(cnt_notna)
        print(start_edge_num)
        assert cnt_notna > start_edge_num, 'start_edge_num should be smaller than the number of non-NA values'
        cur_max_num_edge = min(max_num_edge, cnt_notna)
        print(f'... current max_num_edge: {cur_max_num_edge}')
        for k in range(start_edge_num, cur_max_num_edge+step_size, step_size):
            selected_edges = cur_results.iloc[:k, :].index.tolist()
            all_nodes = { i for t in list(selected_edges) for i in t}
            common_pos_edges = set(selected_edges) & gs_test_pos_set
            common_neg_edges = set(selected_edges) & gs_test_neg_set
            llr = np.log(len(common_pos_edges) / len(common_neg_edges) / (len(gs_test_pos_set) / len(gs_test_neg_set)))
            n_node = len(all_nodes)
            result_dict['k'].append(k)
            result_dict['n'].append(n_node)
            result_dict['llr'].append(llr)
            result_dict['dataset'].append(dataset_name)
            print(f'{dataset_name}, {k}, {n_node}, {llr}')

    llr_ds = pd.DataFrame(result_dict)
    llr_ds.to_csv(output_file, sep='\t', index=False)
    return llr_ds


def compute_llr(feature_df, gs_test_pos_set, gs_test_neg_set):
    pass


def cutoff_prob(model, gs_test, lr_cutoff):
    # the first 2 cols are the ids, the last col is the label
    prob = model.predict_proba(gs_test.iloc[:, 0:-1])
    prob = prob[:, 1]
    pred_df = pd.DataFrame(prob, columns=['prob'])
    pred_df = pd.concat([pred_df, gs_test.iloc[:, -1]], axis=1)
    pred_df = pred_df.sort_values(by='prob', ascending=False)

    P = pred_df['label'].sum()
    N = len(pred_df) - P
    cumulative_pp = np.cumsum(pred_df['label'])
    cumulative_pn = np.arange(len(pred_df)) + 1 - cumulative_pp
    llr_values = np.log((cumulative_pp / cumulative_pn) / (P / N))
    pred_df['llr'] = llr_values

    # find the first prob that has llr >= lr_cutoff
    cutoff = np.log(lr_cutoff)
    for _, row in pred_df[::-1].iterrows():
        if not np.isinf(row['llr']) and row['llr'] >= cutoff:
            cutoff_prob = row['prob']
            break

    return cutoff_prob


def predict_network(predict_results_file, cutoff_p, output_file):
    predicted_df = pd.read_parquet(predict_results_file)
    filtered_df = predicted_df[predicted_df['prediction'] > cutoff_p]
    filtered_df[['P1', 'P2']].to_csv(output_file, sep='\t', index=False)


def predict_all_pairs(model, all_ids, feature_type, cc_dict,
                    mr_dict, extra_feature_file, prediction_dir,
                    output_file, n_jobs=1):
    chunk_size = 1000000
    log.info('Genearating all pairs ...')
    all_ids = sorted(all_ids)
    all_pairs = list(itertools.combinations(all_ids, 2))
    log.info('Genearating all pairs ... done')
    log.info(f'Number of valid ids {format(len(all_ids), ",")}')
    log.info(f'Predicting all {format(len(all_pairs), ",")} pairs ...')

    def process_and_save_chunk(start_idx, chunk_id):
        chunk = all_pairs[start_idx:start_idx + chunk_size]
        chunk_df = pd.DataFrame(chunk, columns=['P1', 'P2'])
        feature_df = extract_features(chunk_df, feature_type, cc_dict,
                                    extra_feature_file, mr_dict)
        predictions = model.predict_proba(feature_df)
        prediction_df = pd.DataFrame(predictions[:, 1], columns=['prediction'])
        prediction_df['P1'] = chunk_df['P1']
        prediction_df['P2'] = chunk_df['P2']
        prediction_df = prediction_df[['P1', 'P2', 'prediction']]
        prediction_df['prediction'] = prediction_df['prediction'].astype('float32')
        table = pa.Table.from_pandas(prediction_df)
        chunk_id = str(chunk_id).zfill(6)
        output_file = f'{prediction_dir}/chunk_{chunk_id}.parquet'
        pq.write_table(table, output_file)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for chunk_id, chunk_start in enumerate(range(0, len(all_pairs), chunk_size)):
            executor.submit(process_and_save_chunk, chunk_start, chunk_id)

    pattern = os.path.join(prediction_dir, 'chunk_*.parquet')
    matching_files = glob.glob(pattern)
    matching_files.sort()
    pq.write_table(pa.concat_tables([pq.read_table(file) for file in matching_files]), output_file)
    for file in matching_files:
        os.remove(file)

    log.info(f'Predicting all {format(len(all_pairs), ",")} pairs ... done.')
