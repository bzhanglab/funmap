import argparse
import sys
import yaml
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from joblib import dump, load
from pathlib import Path
from funmap.funmap import validation_llr, predict_all_pairs, dataset_llr
from funmap.funmap import plot_llr_comparison, explore_data
from funmap.funmap import prepare_features, train_ml_model, prepare_gs_data
from funmap.utils import dict_hash, urls

# add option for user to specify results directory
def arg_parse():
    parser = argparse.ArgumentParser(description='command line arguments.')
    parser.add_argument('-c', '--config-file', required=True, type=str,
                        help='path to experiment configuration yaml file')
    parser.add_argument('-d', '--data-config-file', required=True, type=str,
                        help='path to data configuration yaml file')
    parser.add_argument('-o', '--output-dir', required=False, type=str,
                        help='path to output directory')
    args = parser.parse_args()

    return args


def get_config(cfg_file: str, data_cfg_file: str) -> Tuple[Dict[str, Any],
    Dict[str, Any], Dict[str, Any]]:
    """
    Reads the configuration files and loads the configurations for the run, model, and data.

    Parameters
    ----------
    cfg_file : str
        Path to the run configuration file
    data_cfg_file : str
        Path to the data configuration file

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
    run_cfg['output_edgelist'] = cfg_dict['output_edgelist'] if 'output_edgelist' \
                                in cfg_dict else False

    with open(data_cfg_file, 'r') as stream:
        data_cfg = yaml.load(stream, Loader=yaml.FullLoader)

    return run_cfg, model_cfg, data_cfg


def main():
    args = arg_parse()
    run_cfg, model_cfg, data_cfg = get_config(args.config_file,
                                            args.data_config_file)
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
    output_edgelist = run_cfg['output_edgelist']

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

    # if validation results are available, nothing to do here
    llr_res_file = results_dir / f'llr_results_{max_num_edges}.tsv'
    # llr obtained with each invividual dataset
    llr_dataset_file = results_dir / 'llr_dataset.tsv'

    explore_data(data_cfg, min_sample_count, figure_dir)
    all_feature_df = None
    gs_train = gs_test_pos = gs_test_neg = None

    feature_args = {
        'data_dir': data_dir,
        'data_config': data_cfg,
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
        predicted_all_pairs = pd.read_pickle(predicted_all_pairs_file)
        print(f'Loading predicted all pairs ... done')
    else:
        if ml_model_file.exists():
            print(f'Loading model from {ml_model_file} ...')
            ml_model = load(str(ml_model_file))
            print(f'Loading model ... done')
        else:
            # train an ML model to predict the label
            ml_model = train_ml_model(gs_train, ml_type, seed, n_jobs)
            # save model
            dump(ml_model, ml_model_file, compress=True)

        print('Predicting for all pairs ...')
        predicted_all_pairs = predict_all_pairs(ml_model, all_feature_df,
                                                min_feature_count,
                                                filter_before_prediction,
                                                predicted_all_pairs_file)
        print('Predicting for all pairs ... done.')

    predicted_all_pairs = predicted_all_pairs.astype('float32')
    gs_test_pos_set = set(gs_test_pos.index)
    gs_test_neg_set = set(gs_test_neg.index)

    if not llr_res_file.exists():
        print('Computing LLR with trained model ...')
        validation_llr(all_feature_df, predicted_all_pairs, 'MR',
                    filter_after_prediction, filter_criterion, filter_threshold,
                    filter_blacklist, blacklist_file,
                    max_num_edges, step_size, output_edgelist,
                    gs_test_pos_set, gs_test_neg_set, results_dir)
        print('Done.')
    else:
        llr_res = pd.read_csv(llr_res_file, sep='\t')
    if not llr_dataset_file.exists():
        print('Computing LLR for each dataset ...')
        llr_ds = dataset_llr(all_feature_df, gs_test_pos_set, gs_test_neg_set,
            10000, max_num_edges, 1000, results_dir / 'dataset_llr.tsv')
        print('Done.')
    else:
        llr_ds = pd.read_csv(llr_dataset_file, sep='\t')

    plot_llr_comparison(llr_res, llr_ds, output_file=figure_dir / 'llr_comparison.pdf')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
