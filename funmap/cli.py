import argparse
import sys
import yaml
import os
import json
import tarfile
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import gzip
import pickle
from pathlib import Path
from funmap.funmap import validation_llr, predict_all_pairs, dataset_llr
from funmap.plotting import explore_data, plot_results, merge_and_delete
from funmap.funmap import prepare_features, train_ml_model, prepare_gs_data
from funmap.funmap import feature_mapping
from funmap.utils import dict_hash
from funmap.data_urls import misc_urls as urls
from funmap import __version__

# add option for user to specify results directory
def arg_parse():
    parser = argparse.ArgumentParser(description='command line arguments.')
    parser.add_argument('-c', '--config-file', required=True, type=Path,
                        help='path to experiment configuration yaml file')
    parser.add_argument('-d', '--data-file', required=True, type=Path,
                        help='path to tar gzipped data file')
    parser.add_argument('-o', '--output-dir', required=False, type=str,
                        help='path to output directory')
    parser.add_argument('--version', action='version', version=f'{__version__}')

    args = parser.parse_args()

    return args


def get_config(cfg_file: Path, data_file: Path) -> Tuple[Dict[str, Any],
    Dict[str, Any], Dict[str, Any]]:
    """
    Reads the configuration files and loads the configurations for the run, model, and data.

    Parameters
    ----------
    cfg_file : Path
        Path to the configuration file
    data_file : Path
        Path to the data file

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        A tuple containing the run configuration, model configuration, and data configuration
    """
    run_cfg = {}
    model_cfg = {}
    data_cfg = {}
    with open(cfg_file, 'r') as fh:
        cfg_dict = yaml.load(fh, Loader=yaml.FullLoader)

    # check all file listed under data_files are also in the tar.gz file
    data_files = cfg_dict['data_files']
    # list all the files in the tar.gz file
    with tarfile.open(data_file, "r:gz") as tar:
        tar_files = tar.getnames()
        # get the file names only without the path prefix
        tar_files = [Path(file).name for file in tar_files]

    # check if all files in data_files are in tar_files
    if not all(file['path'] in tar_files for file in data_files):
        print(f'Files listed under data_files are not in the tar.gz file!')
        raise ValueError('Files listed under data_files are not in the tar.gz file!')

    # separte cfg into two parts.
    model_cfg['seed'] = cfg_dict['seed'] if 'seed' in cfg_dict else 42
    model_cfg['cor_type'] = 'spearman'
    model_cfg['split_by'] = 'edge'
    model_cfg['test_size'] = 0.5
    model_cfg['ml_type'] = 'xgboost'
    model_cfg['filter_before_prediction'] = True
    model_cfg['min_feature_count'] = 1
    model_cfg['min_sample_count'] = cfg_dict['min_sample_count'] \
                                    if 'min_sample_count' in cfg_dict else 20
    run_cfg['filter_after_prediction'] = True
    run_cfg['filter_criterion'] = 'max'
    run_cfg['filter_threshold'] = 0.95
    run_cfg['filter_blacklist'] = True
    num_cores = os.cpu_count()
    run_cfg['n_jobs'] = cfg_dict['n_jobs'] if 'n_jobs' in cfg_dict else num_cores
    run_cfg['n_chunks'] = cfg_dict['n_chunks'] if 'n_chunks' in cfg_dict else 4
    run_cfg['max_num_edges'] = cfg_dict['max_num_edges'] if 'max_num_edges' in cfg_dict else 250000
    run_cfg['step_size'] = cfg_dict['step_size'] if 'step_size' in cfg_dict else 100

    data_cfg['dataset_name'] = cfg_dict['dataset_name'] if 'dataset_name' in cfg_dict else 'unknown'
    if 'data_files' not in cfg_dict:
        raise ValueError('data_files not specified in config file')
    data_cfg['data_files'] = cfg_dict['data_files']
    if 'rp_pairs' in cfg_dict:
        data_cfg['rp_pairs'] = cfg_dict['rp_pairs']

    return run_cfg, model_cfg, data_cfg


def main():
    args = arg_parse()
    run_cfg, model_cfg, data_cfg = get_config(args.config_file, args.data_file)
    np.random.seed(model_cfg['seed'])
    model_dir = 'saved_models'
    prediction_dir = 'saved_predictions'
    ml_type = model_cfg['ml_type']
    min_feature_count = model_cfg['min_feature_count']
    min_sample_count = model_cfg['min_sample_count']
    filter_before_prediction = model_cfg['filter_before_prediction']
    test_size = model_cfg['test_size']
    seed = model_cfg['seed']
    cor_type = model_cfg['cor_type']
    split_by = model_cfg['split_by']
    filter_after_prediction = run_cfg['filter_after_prediction']
    filter_criterion = run_cfg['filter_criterion']
    filter_threshold = run_cfg['filter_threshold']
    filter_blacklist = run_cfg['filter_blacklist']
    n_jobs = run_cfg['n_jobs']
    n_chunks = run_cfg['n_chunks']
    max_num_edges = run_cfg['max_num_edges']
    step_size = run_cfg['step_size']

    all_cfg = {**model_cfg, **data_cfg, **run_cfg}
    # results will only be affected by model_cfg and data_cfg
    res_cfg = {**model_cfg, **data_cfg}
    if args.output_dir is None:
        results_dir = Path('results')
        hash_str = dict_hash(res_cfg)
        results_dir = results_dir / f'results-{hash_str}'
    else: # user specified output directory
        results_dir = Path(args.output_dir)
    data_dir = results_dir / 'saved_data'
    model_dir = results_dir / model_dir
    prediction_dir = results_dir / prediction_dir
    figure_dir = results_dir / 'figures'

    print(f'Output directory: {results_dir}')
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # save configuration to results folder
    with open(str(results_dir / 'config.json'), 'w') as fh:
        json.dump(all_cfg, fh, indent=4)

    ml_model_file = model_dir / 'model.pkl.gz'
    predicted_all_pairs_file = prediction_dir / 'predicted_all_pairs.pkl.gz'
    blacklist_file = urls['funmap_blacklist']

    llr_res_files = {feature: results_dir / f'llr_results_{feature}_{max_num_edges}.tsv'
                        for feature in feature_mapping }
    edge_list_paths = {feature: results_dir / 'networks'/ f'network_{feature}_{max_num_edges}.tsv'
                        for feature in feature_mapping }
    # llr obtained with each invividual dataset
    llr_dataset_file = results_dir / 'llr_dataset.tsv'
    all_fig_names = []
    fig_names = explore_data(data_cfg, args.data_file, min_sample_count, figure_dir)
    all_fig_names.extend(fig_names)
    all_feature_df = None
    gs_train = gs_test_pos = gs_test_neg = None

    feature_args = {
        'data_dir': data_dir,
        'data_config': data_cfg,
        'data_file': args.data_file,
        'min_sample_count': min_sample_count,
        'cor_type': cor_type,
        'n_jobs': n_jobs,
        'n_chunks': n_chunks
    }
    all_feature_df, valid_gene_list = prepare_features(**feature_args)
    gs_args = {
        'data_dir': data_dir,
        'all_feature_df': all_feature_df,
        'valid_gene_list': valid_gene_list,
        'min_feature_count': min_feature_count,
        'test_size': test_size,
        'seed': seed,
        'split_by': split_by
    }
    gs_train, gs_test_pos, gs_test_neg = prepare_gs_data(**gs_args)
    # check if models and predictions are available from previous run with the
    # same configuration
    if predicted_all_pairs_file.exists():
        print(f'Loading predicted all pairs from {predicted_all_pairs_file}')
        with gzip.open(predicted_all_pairs_file, 'rb') as fh:
            predicted_all_pairs = pickle.load(fh)
        print(f'Loading predicted all pairs ... done')
    else:
        if ml_model_file.exists():
            print(f'Loading model from {ml_model_file} ...')
            with gzip.open(ml_model_file, 'rb') as fh:
                ml_model = pickle.load(fh)
            print(f'Loading model ... done')
        else:
            ml_model = train_ml_model(gs_train, ml_type, seed, n_jobs)
            with gzip.open(ml_model_file, 'wb') as fh:
                pickle.dump(ml_model, fh)

        print('Predicting for all pairs ...')
        predicted_all_pairs = predict_all_pairs(ml_model, all_feature_df,
                                                min_feature_count,
                                                filter_before_prediction,
                                                predicted_all_pairs_file)
        with gzip.open(predicted_all_pairs_file, 'wb') as fh:
            pickle.dump(predicted_all_pairs, fh)
        print('Predicting for all pairs ... done.')

    for i in predicted_all_pairs:
        predicted_all_pairs[i] = predicted_all_pairs[i].astype('float32')
    gs_test_pos_set = set(gs_test_pos.index)
    gs_test_neg_set = set(gs_test_neg.index)

    # check to see if all files in the llr_res_files list exist
    llr_res_exist = [llr_res_files[f].exists() for f in llr_res_files]
    edge_list_path_exist = [edge_list_paths[f].exists() for f in edge_list_paths]

    if all(llr_res_exist) and all(edge_list_path_exist):
        print('validation results already exist.')
        validation_res = {}
        for ft in feature_mapping:
            validation_res[ft] = {
                'llr_res_path': llr_res_files[ft],
                'edge_list_path': edge_list_paths[ft]
            }
    else:
        print('Computing LLR with trained model ...')
        validation_res = validation_llr(all_feature_df, predicted_all_pairs,
                    filter_after_prediction, filter_criterion, filter_threshold,
                    filter_blacklist, blacklist_file,
                    max_num_edges, step_size, gs_test_pos_set, gs_test_neg_set, results_dir)
        print('Done.')

    if not llr_dataset_file.exists():
        print('Computing LLR for each dataset ...')
        # TODO: adjust the starting number of edges and step size automatically
        llr_ds = dataset_llr(all_feature_df, gs_test_pos_set, gs_test_neg_set,
            10000, max_num_edges, 1000,  llr_dataset_file)
        print('Done.')
    else:
        llr_ds = pd.read_csv(llr_dataset_file, sep='\t')

    fig_names = plot_results(data_cfg, run_cfg, validation_res, llr_ds, gs_train,
                            figure_dir)
    all_fig_names.extend(fig_names)

    merge_and_delete(figure_dir, all_fig_names, 'all_figures.pdf')

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
