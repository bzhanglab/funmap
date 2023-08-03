import click
import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from funmap.funmap import validation_llr, predict_all_pairs, dataset_llr
from funmap.plotting import explore_data, plot_results, merge_and_delete
from funmap.funmap import prepare_features, train_ml_model, prepare_gs_data
from funmap.funmap import feature_mapping, get_funmap
from funmap.data_urls import misc_urls as urls
from funmap.logger import setup_logging, setup_logger
from funmap.utils import setup_experiment
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
@click.option('--data-file', '-d', required=True, type=click.Path(exists=True),
            help='path to tar gzipped data file')
def qc(config_file, data_file):
    """
    Check the data quality.
    """
    # Perform setup tasks here, if any
    setup_logging(config_file)
    run_cfg, model_cfg, data_cfg = setup_experiment(config_file, data_file)

    # Implement qc functionality here
    log.info(f'Checking data quality')
    all_fig_names = []
    figure_dir = Path(run_cfg['results_dir']) / run_cfg['subdirs']['figure_dir']
    min_sample_count = model_cfg['min_sample_count']
    fig_names = explore_data(data_cfg, data_file, min_sample_count, figure_dir)
    all_fig_names.extend(fig_names)
    merge_and_delete(figure_dir, all_fig_names, 'qc.pdf')
    log.info('figure qc.pdf saved to {}'.format(figure_dir))
    log.info('QC complete')


@cli.command(help='run funmap')
@click.option('--config-file', '-c', required=True, type=click.Path(exists=True),
            help='path to experiment configuration yaml file')
@click.option('--data-file', '-d', required=True, type=click.Path(exists=True),
            help='path to tar gzipped data file')
def run(config_file, data_file):
    click.echo(f'Running funmap...')
    setup_logging(config_file)
    run_cfg, model_cfg, data_cfg = setup_experiment(config_file, data_file)

    seed = model_cfg['seed']
    np.random.seed(seed)
    ml_type = model_cfg['ml_type']
    feature_type = model_cfg['feature_type']
    min_feature_count = model_cfg['min_feature_count']
    min_sample_count = model_cfg['min_sample_count']
    filter_before_prediction = model_cfg['filter_before_prediction']
    test_size = model_cfg['test_size']
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
    start_edge_num = run_cfg['start_edge_num']

    results_dir = Path(run_cfg['results_dir'])
    data_dir = results_dir / run_cfg['subdirs']['saved_data_dir']
    model_dir = results_dir / run_cfg['subdirs']['model_dir']
    prediction_dir = results_dir / run_cfg['subdirs']['prediction_dir']
    network_dir = results_dir / run_cfg['subdirs']['network_dir']
    figure_dir = results_dir / run_cfg['subdirs']['figure_dir']

    ml_model_file = model_dir / 'model.pkl.gz'
    predicted_all_pairs_file = prediction_dir / 'predicted_all_pairs.pkl.gz'
    blacklist_file = urls['funmap_blacklist']

    llr_res_files = {feature: results_dir / f'llr_results.tsv'
                        for feature in feature_mapping }
    edge_list_paths = {feature: network_dir/ f'network.tsv'
                        for feature in feature_mapping }
    # llr obtained with each invividual dataset
    llr_dataset_file = results_dir / 'llr_dataset.tsv'
    all_feature_df = None
    gs_train = gs_test_pos = gs_test_neg = None

    feature_args = {
        'data_dir': data_dir,
        'data_config': data_cfg,
        'data_file': data_file,
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
            ml_model = train_ml_model(gs_train, ml_type, feature_type, seed, n_jobs)
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
                    max_num_edges, step_size, gs_test_pos_set,
                    gs_test_neg_set, results_dir, network_dir)
        print('Done.')

    if not llr_dataset_file.exists():
        print('Computing LLR for each dataset ...')
        step_size = int(start_edge_num / 10)
        llr_ds = dataset_llr(all_feature_df, gs_test_pos_set, gs_test_neg_set,
            start_edge_num, step_size, max_num_edges,  llr_dataset_file)
        print('Done.')
    else:
        llr_ds = pd.read_csv(llr_dataset_file, sep='\t')

    get_funmap(validation_res, run_cfg, network_dir)
    all_fig_names = []
    fig_names = plot_results(data_cfg, run_cfg, validation_res, llr_ds, gs_train,
                            figure_dir)
    all_fig_names.extend(fig_names)

    merge_and_delete(figure_dir, all_fig_names, 'results.pdf')


if __name__ == '__main__':
    cli()
