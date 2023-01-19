import argparse
import sys
import yaml
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
import hashlib
from IPython import embed
from joblib import dump, load
from pathlib import Path
from funmap.funmap import validation_llr, predict_all_pairs
from funmap.funmap import prepare_features, train_ml_model, prepare_gs_data
from funmap.utils import get_datafile_path


def arg_parse():
    parser = argparse.ArgumentParser(description='command line arguments.')
    parser.add_argument('-c', '--config-file', required=True,
                        type=argparse.FileType('r'),
                        help='path to experiment configuration yaml file')
    parser.add_argument('-d', '--data-config-file', required=True,
                        type=argparse.FileType('r'),
                        help='path to data configuration yaml file')
    args = parser.parse_args()

    return args


# https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.sha256()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()[:8]


def get_config(cfg_file, data_cfg_file):
    run_cfg = {}
    model_cfg = {}
    data_cfg = {}
    with open(cfg_file, 'r') as stream:
        cfg_dict = yaml.load(stream, Loader=yaml.FullLoader)

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
    results_dir = 'results'
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
    hash_str = dict_hash(res_cfg)
    # save configuration to results folder
    results_prefix = f'results-{hash_str}'
    results_dir = os.path.join(results_dir, results_prefix)
    data_dir = os.path.join(results_dir, 'saved_data')
    model_dir = os.path.join(results_dir, model_dir)
    prediction_dir = os.path.join(results_dir, prediction_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    with open(os.path.join(results_dir, 'config.json'), 'w') as fh:
        json.dump(all_cfg, fh, indent=4)

    ml_model_file = os.path.join(model_dir, 'model.pkl.gz')
    predicted_all_pairs_file = os.path.join(prediction_dir, 'predicted_all_pairs.pkl.gz')
    blacklist_file = get_datafile_path('funmap_blacklist.txt')

    # if validation results are available, nothing to do here
    llr_res_file = os.path.join(results_dir, 'llr_res.tsv')
    if os.path.exists(llr_res_file):
        print(f'{llr_res_file} exists ... nothing to be done')
        return

    all_feature_df = None
    gs_train = gs_test_pos = gs_test_neg = None


    # check if models and predictions are available from previous run with the
    # same configuration
    if os.path.exists(predicted_all_pairs_file):
        print(f'Loading predicted all pairs from {predicted_all_pairs_file}')
        predicted_all_pairs = pd.read_pickle(predicted_all_pairs_file)
        print(f'Loading predicted all pairs ... done')
    else:
        if os.path.exists(ml_model_file):
            print(f'Loading model from {ml_model_file} ...')
            ml_model = load(ml_model_file)
            print(f'Loading model ... done')
        else:
            # train an ML model to predict the label
            cur_args = {
                        'data_dir': data_dir,
                        'data_config': data_cfg,
                        'min_sample_count': min_sample_count,
                        'cor_type': cor_type,
                        'n_jobs': n_jobs,
                        'n_chunks': n_chunks
            }
            all_feature_df, valid_gene_list = prepare_features(**cur_args)
            cur_args = {
                'data_dir': data_dir,
                'all_feature_df': all_feature_df,
                'valid_gene_list': valid_gene_list,
                'min_feature_count': min_feature_count,
                'test_size': test_size,
                'seed': seed,
                'split_by': split_by
            }
            gs_train, gs_test_pos, gs_test_neg = prepare_gs_data(**cur_args)
            ml_model = train_ml_model(gs_train, ml_type, seed, n_jobs)
            # save model
            dump(ml_model, ml_model_file, compress=True)

        print('Predicting for all pairs ...')
        if all_feature_df is None:
            cur_args = {
                        'data_dir': data_dir,
                        'data_config': data_cfg,
                        'min_sample_count': min_sample_count,
                        'cor_type': cor_type,
                        'n_jobs': n_jobs,
                        'n_chunks': n_chunks
            }
            all_feature_df, valid_gene_list = prepare_features(**cur_args)

        predicted_all_pairs = predict_all_pairs(ml_model, all_feature_df,
                                                min_feature_count,
                                                filter_before_prediction,
                                                predicted_all_pairs_file)
        print('Predicting for all pairs ... done.')

    predicted_all_pairs = predicted_all_pairs.astype('float32')

    if all_feature_df is None:
        cur_args = {
            'data_dir': data_dir,
            'data_config': data_cfg,
            'min_sample_count': min_sample_count,
            'cor_type': cor_type,
            'n_jobs': n_jobs,
            'n_chunks': n_chunks
        }
        all_feature_df, valid_gene_list = prepare_features(**cur_args)

    if gs_train is None or gs_test_pos is None or gs_test_neg is None:
        cur_args = {
                'data_dir': data_dir,
                'all_feature_df': all_feature_df,
                'valid_gene_list': valid_gene_list,
                'min_feature_count': min_feature_count,
                'test_size': test_size,
                'seed': seed,
                'split_by': split_by
            }
        gs_train, gs_test_pos, gs_test_neg = prepare_gs_data(**cur_args)

    gs_test_pos_set = set(gs_test_pos.index)
    gs_test_neg_set = set(gs_test_neg.index)
    validation_llr(all_feature_df, predicted_all_pairs, 'MR',
                filter_after_prediction, filter_criterion, filter_threshold,
                filter_blacklist, blacklist_file,
                max_num_edges, step_size, output_edgelist,
                gs_test_pos_set, gs_test_neg_set, results_dir)

    print('Validating ... done.')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
