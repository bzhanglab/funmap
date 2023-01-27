from typing import List, Union, Optional
import os
from tqdm import tqdm
import gc
import itertools
import warnings
import csv
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from joblib import Parallel, delayed
from funmap.utils import chunks
from funmap.utils import get_data_dict
from imblearn.under_sampling import RandomUnderSampler

# file hosted on figshare
REACTOME_GOLD_STANDARD = 'https://tinyurl.com/547hdnzk'

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

    gs_edge_df = pd.read_csv(gs_path, sep='\t')
    gs_edge_df = gs_edge_df.rename(columns={gs_edge_df.columns[0]: 'P1',
                                            gs_edge_df.columns[1]: 'P2'})
    gs_edge_df = gs_edge_df[gs_edge_df['P1'].isin(valid_gene_list) &
                            gs_edge_df['P2'].isin(valid_gene_list) &
                            (gs_edge_df['P1'] != gs_edge_df['P2'])]
    m = ~pd.DataFrame(np.sort(gs_edge_df[['P1','P2']], axis=1)).duplicated()
    gs_edge_df = gs_edge_df[list(m)]
    gs_edge_df.reset_index(drop=True, inplace=True)
    return gs_edge_df


def compute_cc(edges, data_dict, min_valid_count, cor_func):
    """Compute the correlation coefficient for each edge in the list of edges and for each
    feature in the data_dict.

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        A list of edges for which the correlation coefficient will be computed.
    data_dict : Dict[str, pd.DataFrame]
        A dictionary where the keys are feature names and the values
        are dataframes containing the feature values for each gene.
    min_valid_count : int
        The minimum number of valid data points required to compute the correlation coefficient.
    cor_func : Callable
        A function that takes in two arrays and returns the correlation coefficient.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation coefficients
    """
    all_features = [f'{ds}' for ds in data_dict.keys()]
    cor_list = []
    all_indices = []
    for e in edges:
        cur_edge = tuple(sorted(e))
        cur_edge_str = '_'.join(cur_edge)
        values = np.empty([1, len(all_features)])
        value_df = pd.DataFrame(values, index=[cur_edge_str],
                                columns=all_features, dtype=np.float32)
        for feature in all_features:
            cur_data = data_dict[feature]
            if cur_edge[0] in cur_data and cur_edge[1] in cur_data:
                data1 = cur_data.loc[:, cur_edge[0]].values
                data2 = cur_data.loc[:, cur_edge[1]].values
                n_valid = ~np.isnan(data1) & ~np.isnan(data2)
                n_valid_count = np.sum(n_valid)
                if n_valid_count >= min_valid_count:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr, _ = cor_func(data1[n_valid], data2[n_valid])
                        value_df.loc[cur_edge_str, feature] = corr
                else:
                    value_df.loc[cur_edge_str, feature] = np.nan
            else:
                value_df.loc[cur_edge_str, feature] = np.nan

        new_values = value_df.values.reshape(-1)
        cor_list.append(list(new_values))
        all_indices.append(cur_edge)

    new_col_name = [f'{col}_CC' for col in all_features]
    cor_df = pd.DataFrame(cor_list, columns=new_col_name, index=all_indices,
                        dtype=np.float32)
    return cor_df


def extract_gs_features(gs_df: pd.DataFrame,
                all_feature_df: pd.DataFrame):
    """
    Extracts the features from the all_feature_df DataFrame that correspond to the
    pairs of columns in the gs_df DataFrame.

    Parameters
    ----------
    gs_df (pd.DataFrame): DataFrame containing pairs of columns to be used as index.
    all_feature_df (pd.DataFrame): DataFrame containing all features to be extracted.

    Returns
    -------
    pd.DataFrame: DataFrame containing extracted features with index from gs_df.
    """
    cols = gs_df.columns[0:2]
    gs_df.index = [tuple(sorted(x)) for x in zip(gs_df.pop(cols[0]), gs_df.pop(cols[1]))]
    l_tuple = list(gs_df.index)
    feature_df = all_feature_df.loc[l_tuple, :]
    feature_df = pd.merge(gs_df, feature_df, left_index=True, right_index=True)

    return feature_df


def generate_all_pairs(data_config, min_sample_count):
    """
    Generates a list of all valid protein pairs, along with a list of all valid proteins,
    based on the provided data configuration and minimum sample count.

    Parameters
    ----------
    data_config :
        Configuration for the data to be used in the function.
    min_sample_count : int
        The minimum number of valid samples required for a protein to be considered valid.

    Returns
    -------
    tuple of (pandas.DataFrame, list)
        A DataFrame containing all valid protein pairs, with columns 'P1' and 'P2',
        and a list of all valid proteins.
    """
    data_dict = get_data_dict(data_config, min_sample_count)
    all_valid_proteins = set()
    for i in data_dict:
        cur_data = data_dict[i]
        is_valid = cur_data.notna().sum() >= min_sample_count
        valid_count = np.sum(is_valid)
        valid_p = cur_data.columns[is_valid].values
        all_valid_proteins = all_valid_proteins.union(set(valid_p))
        print(f'{i} -- ')
        print(f'  # of samples: {len(cur_data.index)}')
        print(f'  # of proteins: {len(cur_data.columns)}')
        print(f'  # of proteins with at least {min_sample_count} valid samples: {valid_count}')


    all_valid_proteins = list(all_valid_proteins)
    all_valid_proteins.sort()
    # valid protein with at least min_sample_count samples in at least on cancer type
    print(f'total number of valid proteins: {len(all_valid_proteins)}')

    pair_list = []
    for i in range(len(all_valid_proteins)):
        for j in range(i + 1, len(all_valid_proteins)):
            pair_list.append([all_valid_proteins[i], all_valid_proteins[j]])

    df = pd.DataFrame(pair_list, columns=['P1', 'P2'])
    return df, all_valid_proteins


def compute_mr(cor_arr, gene_list):
    """
    Compute the mutual rank of all pairs of genes in the gene list.

    Parameters
    ----------
    cor_arr : numpy.ndarray
        A 1-D array of correlations between all pairs of genes in the gene list.
    gene_list : list
        A list of genes.

    Returns
    -------
    numpy.ndarray
        A 1-D array of mutual ranks of all pairs of genes in the gene list.

    """
    n_genes = len(gene_list)
    res_arr = np.array([1]*len(cor_arr), dtype=np.float32)

    def convert_idx(n, i, j):
        k = (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
        return int(k)

    for i in range(n_genes):
        arr_idx = []
        for j in range(n_genes):
            if i < j:
                arr_idx.append(convert_idx(n_genes, i, j))
            elif i > j:
                arr_idx.append(convert_idx(n_genes, j, i))

        g_cor = cor_arr[arr_idx]
        assert len(g_cor) == n_genes - 1
        n_valid = n_genes - 1 - np.count_nonzero(np.isnan(g_cor))
        tmp = pd.Series(g_cor)
        res = tmp.argsort()
        res.replace(-1, np.nan, inplace=True)
        rank = res.argsort()
        rank.replace(-1, np.nan, inplace=True)
        rank = rank.to_numpy(dtype=np.float32) / n_valid
        del tmp
        gc.collect()
        res_arr[arr_idx] = res_arr[arr_idx] * rank

    res_arr = np.sqrt(res_arr)
    return res_arr


def compute_all_features(edge_df, valid_gene_list, data_config,
                        cor_type, min_sample_count, n_jobs, n_chunks):
    """
    Compute feature dataframe for given edges dataframe and data configuration.

    Parameters:
    -----------
    edge_df : pd.DataFrame
        Dataframe containing edges between genes.
    valid_gene_list : list
        List of genes that are considered valid
    data_config : dict
        Configuration of data sources
    cor_type : str
        Type of correlation to use (only 'spearman' is supported)
    min_sample_count : int
        Minimum sample count required for a gene to be considered
    n_jobs : int
        Number of parallel jobs to use
    n_chunks : int
        Number of chunks to split edges into for processing

    Returns:
    --------
    cor_df : pd.DataFrame
        Dataframe containing computed features for edges.
    """
    assert cor_type == 'spearman', 'correlation type must be spearman'
    cor_func = spearmanr
    data_dict = get_data_dict(data_config, min_sample_count)
    col_name_cc = [f'{ds}_CC' for ds in data_dict.keys()]
    cor_df = pd.DataFrame(columns=col_name_cc)
    all_edges = edge_df.rename(columns={edge_df.columns[0]: 'P1',
                            edge_df.columns[1]: 'P2'})
    all_edges = all_edges.drop_duplicates()
    records = all_edges.to_records(index=False)
    all_edges = list(records)
    n_edges = len(all_edges)
    print(f'# of edges: {n_edges}')
    # to avoid memory error, split the edges into multiple chunks
    print('start computing CC ...')
    chunk_size = len(all_edges) // n_chunks
    if len(all_edges) % n_chunks > 0:
        chunk_size = chunk_size + 1
    for (k, cur_chunk) in enumerate(chunks(all_edges, chunk_size)):
        print(f'processing chunk {k+1} of {n_chunks}')
        job_size = len(cur_chunk) // n_jobs
        if len(cur_chunk) % n_jobs > 0:
            job_size = job_size + 1
        results = Parallel(n_jobs=n_jobs, timeout=99999)(delayed(compute_cc)(edges, data_dict,
                                    min_sample_count, cor_func)
                    for edges in chunks(cur_chunk, job_size))
        for i in range(len(results)):
            # cor_df = cor_df.append(results[i])
            cor_df = pd.concat([cor_df, results[i]], axis=0)
        del results
        gc.collect()
    print('computing CC done')
    # save to temp file to reduce memory ussage
    tmp_cor_file = '/tmp/all_cor_df.fth'
    cor_df.reset_index(inplace=True)
    cor_df.to_feather(tmp_cor_file)
    print('start computing MR ...')
    col_name_mr = [f'{ds}_MR' for ds in data_dict.keys()]
    col_chunks = []
    batch_size = 4
    for chunk in chunks(col_name_mr, batch_size):
        col_chunks.append(chunk)
    start = 0
    res_list = []
    for chunk_idx in range(len(col_chunks)):
        print(f'processing chunk {chunk_idx+1} of {len(col_chunks)}')
        # copy a list of pd.series
        cur_data = []
        cor_df = pd.read_feather(tmp_cor_file)
        cor_df.set_index('index', inplace=True)
        for k in range(len(col_chunks[chunk_idx])):
            cur_data.append(cor_df.iloc[:, start + k].to_numpy(dtype=np.float32))
        del cor_df
        gc.collect()
        results = Parallel(n_jobs=len(col_chunks[chunk_idx]), timeout=99999)(delayed(compute_mr)(cur_data[i], j)
                    for (i, j) in zip(range(len(col_chunks[chunk_idx])),
                                    itertools.repeat(valid_gene_list)))
        print(f'processing chunk {chunk_idx+1} of {len(col_chunks)} ... done')
        for k in range(len(col_chunks[chunk_idx])):
            del cur_data[0]
        for i in range(len(results)):
            res_list.append(results[i])
        start = start + len(col_chunks[chunk_idx])
    print('computing MR done')

    print('merging results ...')
    cor_df = pd.read_feather(tmp_cor_file)
    cor_df['index'] = cor_df['index'].apply(lambda x: tuple(x))
    cor_df.set_index('index', inplace=True)
    col_name_all = col_name_cc.copy()
    col_name_all.extend(col_name_mr)
    mr_df = pd.DataFrame(0, index=cor_df.index, columns=col_name_mr, dtype=np.float32)
    for i in range(len(res_list)):
        mr_df.iloc[:, i] = res_list[i]
    feature_df = pd.DataFrame(0, index=cor_df.index, columns=col_name_all, dtype=np.float32)
    feature_df.iloc[:, :len(col_name_cc)] = cor_df.values
    feature_df.iloc[:, len(col_name_cc):] = mr_df.values
    del mr_df
    del cor_df
    gc.collect()
    os.remove(tmp_cor_file)
    print('merging results ... done')

    return feature_df


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
    model : object
        Trained machine learning model

    """
    X, y = under_sample(data_df)
    assert ml_type == 'xgboost', 'ML model must be xgboost'
    model = train_xgboost_model(X, y, seed, n_jobs)

    return model


def under_sample(df):
    """
    Randomly under-sample the majority class to balance the class distribution.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing the feature set and target variable 'Class'

    Returns
    -------
    X_under: numpy array
        The feature set after under-sampling
    y_under: numpy array
        The target variable after under-sampling
    """

    # impute missing values sample wise
    X = df.drop('Class', axis=1)
    y = df['Class']

    # https://tinyurl.com/2p8zhwaa
    under = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = under.fit_resample(X, y)
    print(f'after sampling, X shape: {X_under.shape}, y shape: {y_under.shape}')
    print(Counter(y_under))
    return X_under, y_under


def feature_type_to_regex(feature_type):
    """
    Convert feature type to regex string.

    Parameters
    ----------
    feature_type : str
        The feature type to be converted. Currently only support 'MR'

    Returns
    -------
    regex_str : str
        The regular expression string corresponding to the feature type.

    """
    assert feature_type == 'MR'
    if feature_type == 'MR':
        regex_str = f'._MR$'

    return regex_str


def train_xgboost_model(X, y, seed, n_jobs):
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
    model : sklearn.model_selection._search.GridSearchCV
        The trained XGBoost model
    """
    model_params = {
        'n_estimators': [50, 150, 250, 300],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    }

    print(f'training xgboost model ...')
    xgb_model = xgb.XGBClassifier(random_state=seed,
                            eval_metric='logloss', n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    clf = GridSearchCV(xgb_model, model_params, scoring='roc_auc', cv=cv,
                    n_jobs=1, verbose=1)
    # use only mutual rank
    feature_type = 'MR'
    regex_str = feature_type_to_regex(feature_type)
    X_sel = X.filter(regex=regex_str)
    print('X_sel shape: ', X_sel.shape)
    print('y shape: ', y.shape)
    model = clf.fit(X_sel, y)
    print(f'training xgboost model ... done')
    return model


def predict_all_pairs(model, all_feature_df, min_feature_count,
                    filter_before_prediction, out_name):
    """
    Predict the probability of all pairs of features in the input dataframe.

    Parameters:
    --------------
    model: object
        trained XGBoost model
    all_feature_df: pandas dataframe
        dataframe containing all feature pairs
    min_feature_count: int
        minimum number of valid feature counts for a pair
    filter_before_prediction: bool
        whether to filter pairs with less than min_feature_count features
        before prediction
    out_name: str
        filename to save the prediction result

    Returns:
    --------------
    pred_df: pandas dataframe
        dataframe containing the prediction result
    """
    if filter_before_prediction:
        all_feature_df = all_feature_df[all_feature_df.iloc[:, 1:].notna().sum(axis=1)
                                >= min_feature_count]

    pred_df = pd.DataFrame(columns=['prediction'], index=all_feature_df.index)

    regex_str = feature_type_to_regex('MR')
    all_feature_df_imp_sel = all_feature_df.filter(regex=regex_str)
    predictions = model.predict_proba(all_feature_df_imp_sel)
    pred_df['prediction'] = predictions[:, 1]

    pred_df.to_pickle(out_name)
    return pred_df


def validation_llr(all_feature_df, predicted_all_pair, feature_type,
                filter_after_prediction, filter_criterion, filter_threshold,
                filter_blacklist, blacklist_file: Path, max_num_edges, step_size,
                output_edge_list, gs_test_pos_set,
                gs_test_neg_set, output_dir: Path):
    """
    Compute Log Likelihood Ratio (LLR) for a given set of edges and a given set of gold
    standard positive and negative edges.
    The function performs filtering and sorting on the input edges before computing LLR.

    Parameters
    ----------
    all_feature_df (pd.DataFrame): Dataframe containing all features
    predicted_all_pair (pd.DataFrame): Dataframe containing predicted values of edges
    feature_type (str): type of feature used to filter edges
    filter_after_prediction (bool): whether to filter edges after prediction
    filter_criterion (str): criterion to filter edges
    filter_threshold (float): threshold value to filter edges
    filter_blacklist (bool): whether to filter edges incident on genes in blacklist
    blacklist_file (Path): file containing blacklist genes
    max_num_edges (int): maximum number of edges to compute LLR for
    step_size (int): step size for iterating over edges
    output_edge_list (Path): path to save selected edges
    gs_test_pos_set (set): set of gold standard positive edges
    gs_test_neg_set (set): set of gold standard negative edges
    output_dir (Path): directory to save LLR results

    Returns
    -------
    llr_res_dict (pd.DataFrame): Dataframe containing LLR results for all selected edges
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # final results
    llr_res_file = output_dir / f'llr_results_{max_num_edges}.tsv'

    if llr_res_file.exists():
        # check if the file can be loaded successfully
        llr_res_dict = pd.read_csv(llr_res_file, sep='\t')
        print(f'{llr_res_file} exists ... nothing to be done')
    else:
        print('Calculating llr_res_dict ...')
        llr_res_dict = {}
        cur_col_name = 'prediction'
        cur_results = predicted_all_pair[[cur_col_name]].copy()
        cur_results.sort_values(by=cur_col_name, ascending=False,
                                inplace=True)
        if filter_after_prediction and feature_type == 'MR':
            if filter_criterion != 'max':
                raise ValueError('Filter criterion must be "max" for MR')
            regex_str = feature_type_to_regex(feature_type)
            all_feature_df_sel = all_feature_df.filter(regex=regex_str)
            all_feature_df_sel = all_feature_df_sel.drop(all_feature_df_sel[all_feature_df_sel.max(axis=1)
                                < filter_threshold].index)
            cur_results = cur_results[cur_results.index.isin(all_feature_df_sel.index)]

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
            # print the smallest values
            # print(f'the smallest values for k = {k}'
                # f'are: {cur_results.iloc[k,:]}')
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

        # write edge list to file
        if output_edge_list:
            print(f'saving edges to file ...')
            out_dir = output_dir / 'networks'
            out_dir.mkdir(parents=True, exist_ok=True)
            edge_list_file_out = out_dir / f'network_{k}.tsv'
            selected_edges = list(cur_results.iloc[:k, :].index)
            with open(edge_list_file_out, 'w') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                for row in list(selected_edges):
                    tsv_writer.writerow(row)

        print('Calculating llr_res_dict ... done')


def load_features(data_config: Path, feature_file: Path,
                pair_file: Path, valid_gene_file: Path, cor_type: str,
                min_sample_count: int, n_jobs: int, n_chunks: int):
    """Load feature data from feather file or compute if not exist.

    Parameters
    ----------
    data_config : Path
        Path to the data configuration file.
    feature_file : Path
        Path to the feather file that contains the feature data.
    pair_file : Path
        Path to the file that contains all pairs of valid genes.
    valid_gene_file : Path
        Path to the file that contains the valid genes.
    cor_type : str
        Type of correlation to use (e.g. 'pearson', 'spearman')
    min_sample_count : int
        Minimum number of samples a gene must be present in to be considered valid.
    n_jobs : int
        Number of parallel jobs to use when computing features.
    n_chunks : int
        Number of chunks to divide the data into for parallel computation.

    Returns
    -------
    all_feature_df : DataFrame
        DataFrame containing the features for all pairs of valid genes.
    valid_gene_list : List[str]
        List of valid genes.
    """
    if feature_file.exists() and valid_gene_file.exists():
            print(f'Loading all features from {feature_file}')
            all_feature_df = pd.read_feather(feature_file)
            all_feature_df['index'] = all_feature_df['index'].apply(lambda x: tuple(x))
            all_feature_df.set_index('index', inplace=True)
            print(f'Loading all features from {feature_file} ... done')
            print(f'Loading all valid gene from {valid_gene_file}')
            with open(valid_gene_file, 'r') as fp:
                valid_genes = fp.read()
                valid_gene_list = valid_genes.split('\n')
                valid_gene_list = valid_gene_list[:-1]
            print(f'Loading all {len(valid_gene_list)} valid gene from {valid_gene_file} ... done')
    else:
        print(f'Computing features for all pairs ...')
        if not pair_file.exists() or not valid_gene_file.exists():
            print(f'Generating all pairs ...')
            edge_df, valid_gene_list = generate_all_pairs(data_config, min_sample_count)
            print(f'Saving all pairs ...')
            edge_df.to_csv(pair_file, sep='\t', header=False, index=False)
            with open(valid_gene_file, 'w') as fp:
                for item in valid_gene_list:
                    # write each item on a new line
                    fp.write(item + '\n')
            print(f'Generating all pairs ... done')
        else:
            print(f'Loading all pairs from {pair_file} ...')
            edge_df = pd.read_csv(pair_file, sep='\t', header=None)
            print(f'Loading all pairs from {pair_file} ... done')
            print(f'Loading all valid gene from {valid_gene_file}')
            with open(valid_gene_file, 'r') as fp:
                valid_genes = fp.read()
                valid_gene_list = valid_genes.split('\n')
                valid_gene_list = valid_gene_list[:-1]
            print(f'Loading all valid gene from {valid_gene_file} ... done')

        all_feature_df = compute_all_features(edge_df, valid_gene_list,
                data_config, cor_type, min_sample_count, n_jobs, n_chunks)
        all_feature_df.reset_index(inplace=True)
        all_feature_df.to_feather(feature_file)
        all_feature_df.set_index('index', inplace=True)
        print(f'Computing feature for all pairs ... done')

    return all_feature_df, valid_gene_list


def prepare_gs_data(**kwargs):
    """
    Prepare gold standard training and test data.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where data files are stored
    all_feature_df : pandas.DataFrame
        Dataframe containing all features
    valid_gene_list : list
        List of valid genes
    test_size : float
        Test size
    seed : int
        Random seed

    Returns
    -------
    gs_train_df : pandas.DataFrame
        Dataframe containing gold standard training data
    gs_test_pos_df : pandas.DataFrame
        Dataframe containing positive gold standard test data
    gs_test_neg_df : pandas.DataFrame
        Dataframe containing negative gold standard test data
    """
    data_dir = kwargs['data_dir']
    all_feature_df = kwargs['all_feature_df']
    valid_gene_list = kwargs['valid_gene_list']
    test_size = kwargs['test_size']
    seed = kwargs['seed']

    gs_train_file = data_dir / 'gold_standard_train.pkl.gz'
    gs_test_pos_file = data_dir / 'gold_standard_test_pos.pkl.gz'
    gs_test_neg_file = data_dir / 'gold_standard_test_neg.pkl.gz'

    if gs_train_file.exists() and gs_test_pos_file.exists() and gs_test_neg_file.exists():
        print(f'Loading existing data file from {gs_train_file}')
        gs_train_df = pd.read_pickle(gs_train_file)
        print(f'Loading existing data file from {gs_train_file} ... done')
        print(f'Loading existing data file from {gs_test_pos_file}')
        gs_test_pos_df = pd.read_pickle(gs_test_pos_file)
        print(f'Loading existing data file from {gs_test_pos_file} ... done')
        print(f'Loading existing data file from {gs_test_neg_file}')
        gs_test_neg_df = pd.read_pickle(gs_test_neg_file)
        print(f'Loading existing data file from {gs_test_neg_file} ... done')
    else:
        print('Preparing gs data ...')
        gs_df = get_valid_gs_data(REACTOME_GOLD_STANDARD, valid_gene_list)
        gs_X_y_train, gs_X_y_test = train_test_split(gs_df,
                                                    test_size=test_size,
                                                    random_state=seed,
                                                    stratify=gs_df[['Class']])
        gs_train_df = extract_gs_features(gs_X_y_train, all_feature_df)
        pd.to_pickle(gs_train_df, gs_train_file)
        cols = gs_X_y_test.columns[0:2]
        gs_X_y_test.index = [tuple(sorted(x)) for x in
                            zip(gs_X_y_test.pop(cols[0]),
                                gs_X_y_test.pop(cols[1]))]

        gs_test_pos_df = gs_X_y_test.loc[gs_X_y_test['Class'] == 1, 'Class']
        gs_test_neg_df = gs_X_y_test.loc[gs_X_y_test['Class'] == 0, 'Class']
        pd.to_pickle(gs_test_pos_df, gs_test_pos_file)
        pd.to_pickle(gs_test_neg_df, gs_test_neg_file)
        print('Preparing gs data ... done')
    return gs_train_df, gs_test_pos_df, gs_test_neg_df


def prepare_features(**kwargs):
    """
    Prepare features for the given dataset and correlation type.

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the data files.
    data_config : Dict
        A dictionary containing the configuration for the dataset.
    min_sample_count : int
        Minimum number of samples required for a gene to be considered valid.
    n_jobs : int
        Number of parallel jobs to run.
    n_chunks : int
        Number of chunks to divide the data into for parallel processing.
    cor_type : str
        Type of correlation to use for computing feature values.

    Returns
    -------
    feature_df : pd.DataFrame
        Dataframe containing the computed features for all pairs of valid genes.
    valid_gene_list : List[str]
        List of valid genes.
    """
    data_dir = kwargs['data_dir']
    data_config = kwargs['data_config']
    min_sample_count = kwargs['min_sample_count']
    n_jobs = kwargs['n_jobs']
    n_chunks = kwargs['n_chunks']
    cor_type = kwargs['cor_type']
    pair_file =data_dir / 'all_pairs.tsv.gz'
    valid_gene_file = data_dir / 'all_valid_gene.txt'
    feature_file = data_dir / 'all_features.fth'
    feature_df, valid_gene_list = load_features(data_config, feature_file, pair_file,
            valid_gene_file, cor_type, min_sample_count, n_jobs, n_chunks)
    return feature_df, valid_gene_list
