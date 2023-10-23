import os
import glob
import math
import h5py
import gzip
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.utils import resample
import itertools
from typing import List
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from funmap.utils import get_data_dict, is_url_scheme, read_csv_with_md5_check
from funmap.data_urls import network_info, misc_urls as urls
from funmap.logger import setup_logger

log = setup_logger(__name__)

def get_valid_gs_data(gs_path: str, valid_gene_list: List[str], md5=None):
    log.info(f'Loading gold standard feature file "{gs_path}" ...')
    if is_url_scheme(gs_path):
        gs_edge_df = read_csv_with_md5_check(gs_path, expected_md5=md5,
                            local_path='download_gs_file', sep='\t')
        if gs_edge_df is None:
            raise ValueError('Failed to download gold standard file')
    else:
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

def assemble_feature_df(h5_file_mapping, df, dataset='cc'):
    df.reset_index(drop=True, inplace=True)
    # Initialize feature_df with columns for HDF5 file keys and 'label'
    file_keys = list(h5_file_mapping.keys())
    feature_df = pd.DataFrame(columns=file_keys + ['label'])

    # Iterate over HDF5 files and load feature values
    for key, file_path in h5_file_mapping.items():
        with h5py.File(file_path, 'r') as h5_file:
            gene_ids = h5_file['ids'][:]
            gene_to_index = {gene.astype(str): idx for idx, gene in enumerate(gene_ids)}

            # Get gene indices for P1 and P2
            p1_indices = np.array([gene_to_index.get(gene, -1) for gene in df.iloc[:, 0]])
            p2_indices = np.array([gene_to_index.get(gene, -1) for gene in df.iloc[:, 1]])

            f_dataset = h5_file[dataset]
            f_values = np.empty(len(df), dtype=float)
            valid_indices = (p1_indices != -1) & (p2_indices != -1)

            linear_indices_func = lambda row_indices, col_indices, n: np.array(col_indices) - np.array(row_indices) + (2*n - np.array(row_indices) + 1) * np.array(row_indices) // 2
            linear_indices = linear_indices_func(p1_indices[valid_indices], p2_indices[valid_indices], len(gene_ids))

            f_values[valid_indices] = f_dataset[:][linear_indices]
            f_values[~valid_indices] = np.nan

            # Add feature values to the feature_df
            feature_df[key] = f_values

    # if the last column is 'label', assign it to feature_df
    if df.columns[-1] == 'label':
        feature_df['label'] = df[df.columns[-1]]
    else:
        # delete the 'label' column from feature_df
        del feature_df['label']

    return feature_df


def extract_features(df, feature_type, cc_dict,  ppi_feature=None, extra_feature=None, mr_dict=None):
    if feature_type == 'mr':
        if not mr_dict:
            raise ValueError('mr dict is empty')

    feature_dict = cc_dict if feature_type == 'cc' else mr_dict
    feature_df = assemble_feature_df(feature_dict, df, feature_type)
    if ppi_feature is not None:
        ppi_dict = {key: set(value) for key, value in ppi_feature.items()}
        for ppi_source, ppi_tuples in ppi_dict.items():
            feature_df[ppi_source] = df.apply(
                lambda row: 1 if (row['P1'], row['P2']) in ppi_tuples else 0, axis=1)

    # TODO: add extra features if provided
    if extra_feature is not None:
        pass

    # move 'label' column to the end of the dataframe if it exists
    if 'label' in feature_df.columns:
        feature_df = feature_df[[col for col in feature_df.columns if col != 'label'] + ['label']]

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

    return ppi_features


def train_ml_model(data_df, ml_type, seed, n_jobs, feature_mapping, model_dir):
    assert ml_type == 'xgboost', 'ML model must be xgboost'
    models = train_model(data_df.iloc[:, :-1], data_df.iloc[:, -1], seed, n_jobs, feature_mapping,
                        model_dir)

    return models


def train_model(X, y, seed, n_jobs, feature_mapping, model_dir):
    model_params = {
        'n_estimators': [50, 150, 250, 300],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    }

    models = {}
    for ft in feature_mapping:
        # use only mutual rank
        log.info(f'Training model for {ft} ...')
        xgb_model = xgb.XGBClassifier(random_state=seed,
                                eval_metric='logloss', n_jobs=n_jobs)
        cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        clf = GridSearchCV(xgb_model, model_params, scoring='roc_auc', cv=cv,
                        n_jobs=1, verbose=2)
        if ft == 'ex':
            # exclude ppi features
            Xtrain = X.loc[:, ~X.columns.str.endswith('_PPI')]
        else:
            Xtrain = X
        model = clf.fit(Xtrain, y)
        models[ft] = model
        ml_model_file = model_dir / f'model_{ft}.pkl.gz'
        with gzip.open(ml_model_file, 'wb') as fh:
            pickle.dump(model, fh)

        log.info(f'Training model for {ft} ... done')

    return models


def compute_llr(predicted_all_pairs, llr_res_file, start_edge_num, max_num_edges, step_size,
                gs_test):
    # make sure max_num_edges is smaller than the number of non-NA values
    assert max_num_edges < np.count_nonzero(~np.isnan(predicted_all_pairs.iloc[:, -1].values)), \
        f'max_num_edges should be smaller than the number of non-NA values'

    cur_col_name = 'prediction'
    cur_results = predicted_all_pairs.nlargest(max_num_edges, cur_col_name)
    selected_edges_all = cur_results[['P1', 'P2']].apply(lambda row: tuple(sorted({row['P1'], row['P2']})), axis=1)

    gs_test_pos_set = set(gs_test[gs_test['label'] == 1][['P1', 'P2']].apply(lambda row: tuple(sorted({row['P1'], row['P2']})), axis=1))
    gs_test_neg_set = set(gs_test[gs_test['label'] == 0][['P1', 'P2']].apply(lambda row: tuple(sorted({row['P1'], row['P2']})), axis=1))
    n_gs_test_pos_set = len(gs_test_pos_set)
    n_gs_test_neg_set = len(gs_test_neg_set)

    result_dict = defaultdict(list)
    # llr_res_dict only save maximum of max_steps data points for downstream
    # analysis / plotting
    total = math.ceil((max_num_edges - start_edge_num) / step_size) + 1
    for k in tqdm(range(start_edge_num, max_num_edges+step_size, step_size), total=total, ascii=' >='):
        selected_edges = set(selected_edges_all[:k])
        all_nodes = set(itertools.chain.from_iterable(selected_edges))
        common_pos_edges = selected_edges & gs_test_pos_set
        common_neg_edges = selected_edges & gs_test_neg_set
        try:
            lr = len(common_pos_edges) / len(common_neg_edges) / (n_gs_test_pos_set / n_gs_test_neg_set)
        except ZeroDivisionError:
            lr = 0
        llr = np.log(lr) if lr > 0 else np.nan
        n_node = len(all_nodes)
        result_dict['k'].append(k)
        result_dict['n'].append(n_node)
        result_dict['llr'].append(llr)

    llr_res = pd.DataFrame(result_dict)
    if llr_res_file is not None:
        llr_res.to_csv(llr_res_file, sep='\t', index=False)

    return llr_res


def prepare_gs_data(**kwargs):
    task = kwargs['task']
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
        gs_file_md5 = urls['reactome_gold_standard_md5']

    # TODO:  use user provided gs_file if it is not None

    log.info('Preparing gold standard data')
    gs_df = get_valid_gs_data(gs_file, valid_id_list, md5=gs_file_md5)
    gs_df_balanced = balance_classes(gs_df, random_state=seed)
    del gs_df
    gs_train, gs_test = train_test_split(gs_df_balanced,
            test_size=test_size, random_state=seed,
            stratify=gs_df_balanced.iloc[:, -1])
    if task == 'protein_func':
        ppi_feature = get_ppi_feature()
    else:
        ppi_feature = None
    gs_train_df = extract_features(gs_train, feature_type, cc_dict, ppi_feature,
                                    extra_feature_file, mr_dict)
    gs_test_df =  extract_features(gs_test, feature_type, cc_dict, ppi_feature,
                                    extra_feature_file, mr_dict)

    # store both the ids with gs_test_df for later use
    # add the first two column of gs_test to gs_test_df at the beginning
    gs_test_df = pd.concat([gs_test.iloc[:, :2], gs_test_df], axis=1)

    log.info('Preparing gs data ... done')
    return gs_train_df, gs_test_df


def extract_dataset_feature(all_pairs, feature_file, feature_type='cc'):
    # convert all_pairs to a dataframe
    df = pd.DataFrame(all_pairs, columns=['P1', 'P2'])
    with h5py.File(feature_file, 'r') as h5_file:
        gene_ids = h5_file['ids'][:]
        gene_to_index = {gene.astype(str): idx for idx, gene in enumerate(gene_ids)}

        # Get gene indices for P1 and P2
        p1_indices = np.array([gene_to_index.get(gene, -1) for gene in df.iloc[:, 0]])
        p2_indices = np.array([gene_to_index.get(gene, -1) for gene in df.iloc[:, 1]])

        f_dataset = h5_file[feature_type]
        f_values = np.empty(len(df), dtype=float)
        valid_indices = (p1_indices != -1) & (p2_indices != -1)

        linear_indices_func = lambda row_indices, col_indices, n: np.array(col_indices) - np.array(row_indices) + (2*n - np.array(row_indices) + 1) * np.array(row_indices) // 2
        linear_indices = linear_indices_func(p1_indices[valid_indices], p2_indices[valid_indices], len(gene_ids))

        f_values[valid_indices] = f_dataset[:][linear_indices]
        f_values[~valid_indices] = np.nan
        # extracted feature is the 'prediction' column
        df['prediction'] = f_values

    return df


def dataset_llr(all_ids, feature_dict, feature_type, gs_test, start_edge_num,
                max_num_edge, step_size, llr_dataset_file):
    llr_ds = pd.DataFrame()
    all_ids_sorted = sorted(all_ids)
    all_pairs = list(itertools.combinations(all_ids_sorted, 2))
    all_ds_pred = None

    for dataset in feature_dict:
        log.info(f'Calculating llr for {dataset} ...')
        feature_file = feature_dict[dataset]
        predicted_all_pairs = extract_dataset_feature(all_pairs, feature_file, feature_type)
        if all_ds_pred is None:
            all_ds_pred = predicted_all_pairs['prediction'].values
        else:
            all_ds_pred = np.vstack((all_ds_pred, predicted_all_pairs['prediction'].values))

        cur_llr_res = compute_llr(predicted_all_pairs, None, start_edge_num,
                                max_num_edge, step_size, gs_test)
        cur_llr_res['dataset'] = dataset
        llr_ds = pd.concat([llr_ds, cur_llr_res], axis=0, ignore_index=True)
        llr_ds.to_csv(llr_dataset_file, sep='\t', index=False)
        log.info(f'Calculating llr for {dataset} ... done')

    # calculate llr for all datasets based on the average prediction
    log.info(f'Calculating llr for all datasets average ...')
    all_ds_pred_df = pd.DataFrame(all_pairs, columns=['P1', 'P2'])
    all_ds_pred_avg = np.nanmean(all_ds_pred, axis=0)
    all_ds_pred_df['prediction'] = all_ds_pred_avg
    cur_llr_res = compute_llr(all_ds_pred_df, None, start_edge_num, max_num_edge, step_size, gs_test)
    cur_llr_res['dataset'] = 'all_average'
    llr_ds = pd.concat([llr_ds, cur_llr_res], axis=0, ignore_index=True)
    log.info(f'Calculating llr for all datasets average ... done')
    llr_ds.to_csv(llr_dataset_file, sep='\t', index=False)

    return llr_ds


def get_cutoff(model_dict, gs_test, lr_cutoff):
    cutoff_p_dict = {}
    cutoff_llr_dict = {}
    for ft in model_dict:
        log.info(f'Calculating cutoff prob for {ft} ...')
        model = model_dict[ft]
        if ft == 'ex':
            gs_test_df = gs_test.loc[:, ~gs_test.columns.str.endswith('_PPI')]
        else:
            gs_test_df = gs_test
        prob = model.predict_proba(gs_test_df.iloc[:, 2:-1])
        prob = prob[:, 1]
        pred_df = pd.DataFrame(prob, columns=['prob'])
        pred_df = pd.concat([pred_df, gs_test_df.iloc[:, -1]], axis=1)
        pred_df = pred_df.sort_values(by='prob', ascending=False)

        P = pred_df['label'].sum()
        N = len(pred_df) - P
        cumulative_pp = np.cumsum(pred_df['label'])
        cumulative_pn = np.arange(len(pred_df)) + 1 - cumulative_pp
        llr_values = np.log((cumulative_pp / cumulative_pn) / (P / N))
        pred_df['llr'] = llr_values

        # find the first prob that has llr >= lr_cutoff
        cutoff = np.log(lr_cutoff)
        cutoff_prob = None
        for _, row in pred_df[::-1].iterrows():
            if not np.isinf(row['llr']) and row['llr'] >= cutoff:
                cutoff_prob = row['prob']
                cutoff_llr = row['llr']
                break

        # if cutoff_prob is None, it means that the lr_cutoff is too high
        # and we cannot find a cutoff prob that has llr >= lr_cutoff
        # in this case, we set cutoff_llr be the largest llr value
        if cutoff_prob is None:
            log.warn(f'Cannot find cutoff prob for {ft}, setting lr_cutoff to the largest llr')
            cutoff_llr = pred_df['llr'].max()
            cutoff_prob = pred_df[pred_df['llr'] == cutoff_llr]['prob'].values[0]

        cutoff_p_dict[ft] = cutoff_prob
        cutoff_llr_dict[ft] = cutoff_llr

    return cutoff_p_dict, cutoff_llr_dict


def predict_network(predict_results_file, cutoff_p, output_file):
    for ft in predict_results_file:
        log.info(f'Predicting network for {ft} ...')
        predicted_df = pd.read_parquet(predict_results_file[ft])
        filtered_df = predicted_df[predicted_df['prediction'] > cutoff_p[ft]]
        filtered_df[['P1', 'P2']].to_csv(output_file[ft], sep='\t', index=False, header=None)
        num_edges = len(filtered_df)
        num_nodes = len(set(filtered_df['P1']) | set(filtered_df['P2']))
        log.info(f'Number of edges: {num_edges}')
        log.info(f'Number of nodes: {num_nodes}')
        log.info(f'Predicting network for {ft} ... done')


def predict_all_pairs(model_dict, all_ids, feature_type, ppi_feature, cc_dict,
                    mr_dict, extra_feature_file, prediction_dir,
                    output_file, n_jobs=1):
    chunk_size = 1000000
    log.info('Genearating all pairs ...')
    all_ids = sorted(all_ids)
    all_pairs = list(itertools.combinations(all_ids, 2))
    log.info('Genearating all pairs ... done')
    log.info(f'Number of valid ids {format(len(all_ids), ",")}')
    # remove all "chunk_*.parquet" files in prediction_dir if they exist
    pattern = os.path.join(prediction_dir, 'chunk_*.parquet')
    matching_files = glob.glob(pattern)
    for file in matching_files:
        os.remove(file)

    for ft in model_dict:
        log.info(f'Predicting all pairs ({format(len(all_pairs), ",")}) for {ft} ...')
        model = model_dict[ft]
        def process_and_save_chunk(start_idx, chunk_id):
            chunk = all_pairs[start_idx:start_idx + chunk_size]
            chunk_df = pd.DataFrame(chunk, columns=['P1', 'P2'])
            if ft == 'ex':
                cur_ppi_feature = None
            else:
                cur_ppi_feature = ppi_feature
            feature_df = extract_features(chunk_df, feature_type, cc_dict, cur_ppi_feature,
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
        pq.write_table(pa.concat_tables([pq.read_table(file) for file in matching_files]), output_file[ft])
        for file in matching_files:
            os.remove(file)

        log.info(f'Predicting all {format(len(all_pairs), ",")} pairs for {ft} done.')
