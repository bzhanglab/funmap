import os
import click
import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from funmap.funmap import compute_llr, predict_all_pairs, dataset_llr, predict_all_pairs
from funmap.plotting import explore_data, plot_results, merge_and_delete
from funmap.funmap import  train_ml_model, prepare_gs_data, get_cutoff, get_ppi_feature
from funmap.funmap import compute_features, predict_network
from funmap.data_urls import misc_urls as urls
from funmap.logger import setup_logging, setup_logger
from funmap.utils import setup_experiment, cleanup_experiment, check_gold_standard_file
from funmap.utils import check_extra_feature_file
from funmap import __version__

log = setup_logger(__name__)

@click.group(help='funmap command line interface')
@click.version_option(version=f'{__version__}')
def cli():
    """
    Command line interface for funmap.
    """
    click.echo("====== funmap =======")

@cli.command(help='check the data quality')
@click.option('--config-file', '-c', required=True, type=click.Path(exists=True),
            help='path to experiment configuration yaml file')
@click.option('--force-rerun', '-f', is_flag=True, default=False,
            help='if set, will remove results from previous run first')
def qc(config_file, force_rerun):
    if force_rerun:
        while True:
            confirmation = input("Do you want to remove results from previous run? (y/n): ")
            if confirmation.lower() == 'y':
                click.echo('Removing results from previous run')
                cleanup_experiment(config_file)
                break
            elif confirmation.lower() == 'n':
                click.echo('Not removing results from previous run')
                break
            else:
                click.echo("Invalid input. Please enter 'y' or 'n'.")

    setup_logging(config_file)
    log.info(f'Running QC...')
    cfg = setup_experiment(config_file)
    all_fig_names = []
    figure_dir = Path(cfg['results_dir']) / cfg['subdirs']['figure_dir']
    min_sample_count = cfg['min_sample_count']
    fig_names = explore_data(cfg, min_sample_count, figure_dir)
    all_fig_names.extend(fig_names)
    merge_and_delete(figure_dir, all_fig_names, 'qc.pdf')
    log.info('figure qc.pdf saved to {}'.format(figure_dir))
    log.info('QC complete')


@cli.command(help='run funmap')
@click.option('--config-file', '-c', required=True, type=click.Path(exists=True),
            help='path to experiment configuration yaml file')
@click.option('--force-rerun', '-f', is_flag=True, default=False,
            help='if set, will remove results from previous run first')
def run(config_file, force_rerun):
    click.echo(f'Running funmap...')
    if force_rerun:
        while True:
            confirmation = input("Do you want to remove results from previous run? (y/n): ")
            if confirmation.lower() == 'y':
                click.echo('Removing results from previous run')
                cleanup_experiment(config_file)
                break
            elif confirmation.lower() == 'n':
                click.echo('Not removing results from previous run')
                break
            else:
                click.echo("Invalid input. Please enter 'y' or 'n'.")

    setup_logging(config_file)
    cfg = setup_experiment(config_file)
    extra_feature_file = cfg['extra_feature_file']
    if (extra_feature_file is not None) and (not check_extra_feature_file(extra_feature_file)):
        return
    gs_file = cfg['gs_file']
    if (gs_file is not None) and (not check_gold_standard_file(gs_file)):
        return

    task = cfg['task']
    seed = cfg['seed']
    np.random.seed(seed)
    ml_type = cfg['ml_type']
    feature_type = cfg['feature_type']
    # min_feature_count = cfg['min_feature_count']
    min_sample_count = cfg['min_sample_count']
    # filter_before_prediction = cfg['filter_before_prediction']
    test_size = cfg['test_size']
    # filter_after_prediction = cfg['filter_after_prediction']
    # filter_criterion = cfg['filter_criterion']
    # filter_threshold = cfg['filter_threshold']
    # filter_blacklist = cfg['filter_blacklist']
    n_jobs = cfg['n_jobs']
    lr_cutoff = cfg['lr_cutoff']
    max_num_edges = cfg['max_num_edges']
    step_size = cfg['step_size']
    start_edge_num = cfg['start_edge_num']

    results_dir = Path(cfg['results_dir'])
    saved_data_dir = results_dir / cfg['subdirs']['saved_data_dir']
    model_dir = results_dir / cfg['subdirs']['saved_model_dir']
    prediction_dir = results_dir / cfg['subdirs']['saved_predictions_dir']
    network_dir = results_dir / cfg['subdirs']['network_dir']
    figure_dir = results_dir / cfg['subdirs']['figure_dir']

    if cfg['task'] == 'protein_func':
        feature_mapping = ['ex', 'ei']
    else:
        feature_mapping = ['ex']
        # here the file stored a dictionary of ml models
    ml_model_file = {feature: model_dir / f'model_{feature}.pkl.gz'
                    for feature in feature_mapping }
    predicted_all_pairs_file = {feature: prediction_dir / f'predicted_all_pairs_{feature}.parquet'
                    for feature in feature_mapping }
    llr_res_file = {feature: results_dir / f'llr_results_{feature}.tsv'
                    for feature in feature_mapping }
    edge_list_file = {feature: network_dir/ f'funmap_{feature}.tsv'
                    for feature in feature_mapping }
    # gold standard data include specified feature (cc or mr) and ppi feature (if applicable)
    # and extra feature if applicable
    gs_df_file = saved_data_dir / 'gold_standard_data.h5'
    # blacklist_file = urls['funmap_blacklist']
    # llr obtained with each invividual dataset
    llr_dataset_file = results_dir / 'llr_dataset.tsv'
    gs_train = gs_test = None
    cutoff_p = cutoff_llr = None
    ml_model_dict = {}

    # compute and save cc, mr results
    cc_dict, mr_dict, all_valid_ids = compute_features(cfg, feature_type, min_sample_count,
                                                    saved_data_dir)
    gs_args = {
        'task': task,
        'saved_data_dir': saved_data_dir,
        'cc_dict': cc_dict,
        'mr_dict': mr_dict,
        'feature_type': feature_type,
        'gs_file': gs_file,
        'extra_feature_file': extra_feature_file,
        'valid_id_list': all_valid_ids,
        'test_size': test_size,
        'seed': seed
    }

    all_edge_list_exist = all(os.path.exists(file_path) for file_path in edge_list_file.values())

    if all_edge_list_exist:
        log.info('Fumap network(s) already exists. Skipping model training and prediction.')
    else:
        all_model_exist = all(os.path.exists(file_path) for file_path in ml_model_file.values())
        if all_model_exist:
            log.info(f'Trained model(s) exists. Loading model(s) ...')
            ml_model_dict = {}
            # feature: ex or ei
            for feature in ml_model_file:
                with gzip.open(ml_model_file[feature], 'rb') as fh:
                    ml_model = pickle.load(fh)
                    ml_model_dict[feature] = ml_model
            log.info(f'Loading model(s) ... done')
            if not gs_df_file.exists():
                log.error(f'Trained models found but gold standard data file {gs_df_file} '
                        f'does not exist.')
                return
            with pd.HDFStore(gs_df_file, mode='r') as store:
                gs_train = store['train']
                gs_test = store['test']
        else:
            gs_train, gs_test = prepare_gs_data(**gs_args)
            with pd.HDFStore(gs_df_file, mode='w') as store:
                store.put('train', gs_train)
                store.put('test', gs_test)
            ml_model_dict = train_ml_model(gs_train, ml_type, seed, n_jobs, feature_mapping,
                                        model_dir)

        all_predicted_all_pairs_exist = all(os.path.exists(file_path) for file_path in
                                            predicted_all_pairs_file.values())
        if all_predicted_all_pairs_exist:
            log.info('Predicted all pairs already exists. Skipping prediction.')
        else:
            log.info('Predicting all pairs ...')
            if task == 'protein_func':
                ppi_feature = get_ppi_feature()
            else:
                ppi_feature = None
            pred_all_pairs_args = {
                'model_dict': ml_model_dict,
                'all_ids': all_valid_ids,
                'feature_type': feature_type,
                'ppi_feature': ppi_feature,
                'cc_dict': cc_dict,
                'mr_dict': mr_dict,
                'extra_feature_file': extra_feature_file,
                'prediction_dir': prediction_dir,
                'output_file': predicted_all_pairs_file,
                'n_jobs': n_jobs
            }
            predict_all_pairs(**pred_all_pairs_args)
            log.info('Predicting all pairs ... done')

        cutoff_p, cutoff_llr = get_cutoff(ml_model_dict, gs_test, lr_cutoff)
        log.info(f'cutoff probability: {cutoff_p}')
        log.info(f'cutoff llr: {cutoff_llr}')
        predict_network(predicted_all_pairs_file, cutoff_p, edge_list_file)

    if gs_test is None:
        with pd.HDFStore(gs_df_file, mode='r') as store:
                gs_train = store['train']
                gs_test = store['test']

    all_llr_res_exist = all(os.path.exists(file_path) for file_path in llr_res_file.values())
    all_edge_list_exist = all(os.path.exists(file_path) for file_path in edge_list_file.values())
    if all_llr_res_exist and all_edge_list_exist:
        log.info('LLR results already exist.')
    else:
        for ft in feature_mapping:
            log.info(f'Computing LLR for {ft} ...')
            if not predicted_all_pairs_file[ft].exists():
                log.error(f'Predicted all pairs file {predicted_all_pairs_file[ft]} does not exist.')
                return
            predicted_all_pairs = pd.read_parquet(predicted_all_pairs_file[ft])
            # also save the llr results to file
            compute_llr(predicted_all_pairs, llr_res_file[ft], start_edge_num, max_num_edges, step_size,
                        gs_test)
            log.info(f'Computing LLR for {ft} ... done')

    validation_res = {}
    for ft in feature_mapping:
        validation_res[ft] = {
            'llr_res_path': llr_res_file[ft],
            'edge_list_path': edge_list_file[ft]
        }
    if not llr_dataset_file.exists():
        log.info('Computing LLR for each dataset ...')
        # feature_dict = cc_dict if feature_type == 'cc' else mr_dict
        # use CC features for individual dataset
        llr_ds = dataset_llr(all_valid_ids, cc_dict, 'cc', gs_test, start_edge_num,
                            max_num_edges, step_size, llr_dataset_file)
        log.info('Done.')
    else:
        llr_ds = pd.read_csv(llr_dataset_file, sep='\t')

    if not ml_model_dict:
        log.info(f'Trained model(s) exists. Loading model(s) ...')
        for feature in ml_model_file:
            with gzip.open(ml_model_file[feature], 'rb') as fh:
                ml_model = pickle.load(fh)
                ml_model_dict[feature] = ml_model
        log.info(f'Loading model(s) ... done')

    all_fig_names = []
    if cutoff_llr is None:
        cutoff_p, cutoff_llr = get_cutoff(ml_model_dict, gs_test, lr_cutoff)

    gs_dict = {}
    gs_dict[feature_type.upper()] = gs_train
    if task == 'protein_func' and feature_type.upper() == 'MR' and 'rp_pairs' in cfg:
        # extract gs data for CC and MR for plotting
        gs_args = {
                'task': task,
                'saved_data_dir': saved_data_dir,
                'cc_dict': cc_dict,
                'mr_dict': mr_dict,
                'feature_type': 'cc',
                'gs_file': gs_file,
                # no extra feature for plotting
                'extra_feature_file': None,
                'valid_id_list': all_valid_ids,
                'test_size': test_size,
                'seed': seed
            }
        gs_train, gs_test = prepare_gs_data(**gs_args)
        gs_dict['CC'] = gs_train
        pass

    fig_names = plot_results(cfg, validation_res, llr_ds, gs_dict, cutoff_llr,
                            figure_dir)
    all_fig_names.extend(fig_names)

    merge_and_delete(figure_dir, all_fig_names, 'results.pdf')


if __name__ == '__main__':
    cli()
