from typing import List, Dict
import os
from pathlib import Path
import pandas as pd
import numpy as np
from funmap.utils import get_data_dict, get_node_edge_overlap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import seaborn as sns
import PyPDF2
from matplotlib_venn import venn2, venn2_circles
import networkx as nx
import powerlaw


def edge_number(x, pos):
    """
    Formatter function to format the x-axis tick labels

    Parameters
    ----------
    x : float
        The value to be formatted.
    pos : float
        The tick position.

    Returns
    -------
    s : str
        The formatted string of the value.
    """
    if x >= 1e6:
        s = '{:1.1f}M'.format(x*1e-6)
    elif x == 0:
        s = '0'
    else:
        s = '{:1.0f}K'.format(x*1e-3)
    return s


def plot_llr_comparison(validation_results, llr_ds, output_file='llr_comparison.pdf'):
    """
    Plot the comparison of log likelihood ratios (LLR) based on model prediction
    using all datasets and LLR results for each dataset.

    Parameters
    ----------
    validation_results : dict
        a dictionary containing the validation results
    llr_ds : pandas DataFrame
        LLR results for each dataset
    output_file : str/Path, optional
        Output file name or path for saving the plot, by default 'llr_comparison.pdf'

    Returns
    -------
    None

    """
    datasets = sorted(llr_ds['dataset'].unique().tolist())
    fig, ax = plt.subplots(figsize=(20, 16))

    start = -1
    for ds in datasets:
        cur_df = llr_ds[llr_ds['dataset'] == ds]
        ax.plot(cur_df['k'], cur_df['llr'], label=ds)
        if start == -1:
            start = cur_df['k'].iloc[0]
    # plot llr_res with the same start point
    for ft in validation_results:
        llr_res = pd.read_csv(validation_results[ft]['llr_res_path'], sep='\t')
        llr_res = llr_res[llr_res['k'] >= start]
        ax.plot(llr_res['k'], llr_res['llr'], label=f'funmap_{ft}', linewidth=3)

    ax.xaxis.set_major_formatter(edge_number)
    ax.set_xlabel('number of gene pairs', fontsize=16)
    ax.set_ylabel('log likelihood ratio', fontsize=16)
    ax.yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.box(on=None)
    plt.savefig(output_file, bbox_inches='tight')
    return output_file


def explore_data(data_config: Path,
                data_file: Path,
                min_sample_count: int,
                output_dir: Path):
    """
    Generate plots to explore and visualize data

    Parameters
    ----------
    data_config: Path
        Path to the data configuration file
    data_file: Path
        Path to the data file
    min_sample_count: int
        The minimum number of samples required to consider a dataset
    output_dir: Path
        The directory to save the output plots

    Returns
    -------
    A list of file names of the generated plots

    """
    print('Generating plots to explore and visualize data ...')
    data_dict = get_data_dict(data_config, data_file, min_sample_count)
    fig_names = []

    # sample wise median expression plot for each dataset
    data = []
    data_keys = []

    max_col_to_plot = 100

    for ds in data_dict:
        data_df = data_dict[ds]
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        cur_data = data_df.T
        cur_data.dropna(inplace=True)
        if cur_data.shape[1] > max_col_to_plot:
            cur_data = cur_data.sample(max_col_to_plot, axis=1)
        ax[0].boxplot(cur_data)
        ax[0].set_ylabel('expression')
        if data_df.shape[0] > max_col_to_plot:
            ax[0].set_xlabel(f'random selected {max_col_to_plot} samples (total n={data_df.shape[0]})')
            ax[0].set_xticklabels([])
            ax[0].set_xticks([])
        else:
            ax[0].set_xlabel('sample')
            ax[0].set_xticklabels(cur_data.columns, rotation=45, ha='right')

        # density plot for each sample in each dataset
        for i in range(data_df.shape[0]):
            sns.kdeplot(data_df.iloc[i, :], linewidth=1, ax=ax[1])
        locator=MaxNLocator(60)
        ax[1].xaxis.set_major_locator(locator)
        ax[1].set_xlabel('values')
        ax[1].set_ylabel('density')
        ticks_loc = ax[1].get_xticks().tolist()
        ax[1].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')

        # set title for the figu
        fig.suptitle(f'{ds}', fontsize=16)
        fig.tight_layout()
        cur_file_name = f'{ds}_sample_plot.pdf'
        fig_names.append(cur_file_name)
        fig.savefig(output_dir / cur_file_name)
        data_keys.append(ds)
        data.append(data_df.median(axis=1).values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data)
    ax.set_xticklabels(data_keys, rotation=45)
    ax.yaxis.grid(color = 'gainsboro', linestyle = 'dotted')

    ax.set_xlabel('dataset')
    ax.set_ylabel('median expression')
    plt.box(on=None)
    fig.tight_layout()
    file_name = 'data_box_plot.pdf'
    fig_names.append(file_name)
    fig.savefig(output_dir / file_name)

    # boxplot of the number of samples and genes
    sample_count = pd.DataFrame(
        {'count':[data_dict[ds].shape[0] for ds in data_dict],
        'dataset': [ds for ds in data_dict]})
    gene_count = pd.DataFrame({
        'count':[data_dict[ds].shape[1] for ds in data_dict],
        'dataset': [ds for ds in data_dict]})

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    bars0 = ax[0].barh(sample_count['dataset'], sample_count['count'], color='#774FA0')
    bars1 = ax[1].barh(gene_count['dataset'], gene_count['count'], color='#7DC462')

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[0].bar_label(bars0, label_type='edge', fontsize=10)
    ax[0].set_xlabel('number of samples')

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_yticklabels([])
    ax[1].tick_params(axis='x', which='major', labelsize=12)
    ax[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax[1].bar_label(bars1, label_type='edge', fontsize=10)
    ax[1].set_xlabel('number of genes')

    fig.tight_layout()
    file_name = 'sample_gene_count.pdf'
    fig.savefig(output_dir / file_name)
    fig_names.append(file_name)
    return fig_names


def plot_results(data_cfg: dict, run_cfg: dict, validation_results: dict,
                llr_ds: pd.DataFrame, gs_train: pd.DataFrame,
                figure_dir: Path) -> List[str]:
    """
    Plot the results of the analysis.

    Parameters
    ----------
    data_cfg : dict
        A dictionary containing the configuration for the data being analyzed.
    run_cfg : dict
        A dictionary containing the configuration for the analysis.
    validation_results : dict
        A dictionary containing the results of the validation.
    llr_ds : pandas.DataFrame
        The log-likelihood ratio for the dataset.
    gs_train : pandas.DataFrame
        The training data for the analysis.
    figure_dir : Path
        The directory where the figures will be saved.

    Returns
    -------
    fig_names : List[str]
        A list of names of the figures generated by the analysis.
    """
    fig_names = []
    file_name = 'llr_compare_dataset.pdf'
    plot_llr_comparison(validation_results, llr_ds, output_file=figure_dir / file_name)
    fig_names.append(file_name)

    if 'rp_pairs' in data_cfg:
        file_names = plot_pair_llr(gs_train, output_dir=figure_dir, rp_pairs=data_cfg['rp_pairs'])
        fig_names.extend(file_names)

    # note that the cutoff is for LR, not LLR (log of LR)
    if 'cutoff' in run_cfg:
        cutoff = run_cfg['cutoff']
    else:
        cutoff = 50
    file_name = 'llr_compare_networks.pdf'
    n_edge_dict = plot_llr_compare_networks(validation_results, cutoff, output_file=figure_dir / file_name)
    fig_names.append(file_name)

    # the information about other networks is fixed for now
    other_network_info = {
        'name': ['BioGRID', 'BioPlex', 'HI-union', 'STRING'],
        'type': ['BioGRID', 'BioPlex', 'HI', 'STRING'],
        'url': ['https://figshare.com/ndownloader/files/39125054',
                'https://figshare.com/ndownloader/files/39125051',
                'https://figshare.com/ndownloader/files/39125093',
                'https://figshare.com/ndownloader/files/39125090'
            ]
    }
    # convert the info to a data frame where the url is read as a dataframe
    network_info = pd.DataFrame(other_network_info)
    network_info['el'] = network_info['url'].apply(lambda x: pd.read_csv(x,
                                                    sep='\t', header=None))
    network_info = network_info.drop(columns=['url'])

    # for each funmap, create a dataframe
    for ft in validation_results:
        edge_file_path = validation_results[ft]['edge_list_path']
        funmap_el = pd.read_csv(edge_file_path, sep='\t', header=None)
        n_edge = n_edge_dict[ft]
        funmap_el = funmap_el.iloc[:n_edge, :]

        funmap_df = pd.DataFrame({'name': ['FunMap'], 'type': ['FunMap'], 'el': [funmap_el]})
        all_network_info = pd.concat([network_info, funmap_df], ignore_index=True)
        overlap_info = get_node_edge_overlap(all_network_info)
        node_color, edge_color = '#7DC462', '#774FA0'
        for (type, color) in zip(['node', 'edge'], [node_color, edge_color]):
            fig_name = plot_overlap_venn(f'funmap_{ft}', overlap_info[type], type, color, figure_dir)
            fig_names.append(fig_name)

        fig_name = plot_network_stats(all_network_info, ft, figure_dir)
        fig_names.append(fig_name)

    return fig_names


def plot_1d_llr(ax, feature_df, feature_name, feature_type, data_type, n_bins):
    """
    Plot the 1D histogram of the likelihood ratio for each feature

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot where the histogram is to be plotted.
    feature_df : pd.DataFrame
        DataFrame containing all features and their values.
    feature_name : str
        The name of the feature for which histogram is to be plotted.
    feature_type : str
        The type of the feature, either 'CC' or 'MR'.
    data_type : str
        The type of data, either 'RNA' or 'PRO'.
    n_bins : int
        The number of bins for the histogram.

    Returns
    -------
    None
    """
    df = feature_df.loc[:, [feature_name]]
    cur_df = df.dropna()
    cur_df_vals = cur_df.values.reshape(-1)
    clr = '#bcbddc'
    data_range = {'CC': (-1, 1), 'MR': (0, 1)}
    if data_type == 'PRO':
        ax.hist(cur_df_vals, bins=n_bins, range=data_range[feature_type], color=clr,
            orientation='horizontal',
            density=True)
        ax.text(0.95, 0.95, data_type,
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            rotation=-90,
            color='black', fontsize=16)
        ax.set_xlim(0,2.5)
    else:
        ax.hist(cur_df_vals, bins=n_bins, range=data_range[feature_type], color=clr,
            density=True)
        ax.text(0.02, 0.9, data_type,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=16)
        ax.set_ylabel('density', fontsize=16)
        ax.set_ylim(0,2.5)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def plot_2d_llr(ax, feature_df, feature_type, pair_name, rna_feature, pro_feature, n_bins):
    """
    Plots a 2D log likelihood ratio between two features in a scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot to plot the log likelihood ratio on.
    feature_df : pandas.DataFrame
        DataFrame containing all features and target variables.
    feature_type : str
        Type of feature, either "CC" (correlation coefficient) or "MR" (mutual rank).
    pair_name : str
        Name of the feature pair.
    rna_feature : str
        Name of the RNA feature in `feature_df`.
    pro_feature : str
        Name of the protein feature in `feature_df`.
    n_bins : int
        Number of bins in the 2D histogram.

    Returns
    -------
    fig : matplotlib.collections.QuadMesh
        The mesh plot of the log likelihood ratio.

    """
    data_types = ['RNA', 'PRO']
    label_mapping = {'PRO': 'Protein', 'RNA': 'mRNA'}
    feature_label_mapping = {'CC': 'correlation coefficient',
                            'MR': 'mutual rank'}
    cnt = {}
    cnt_pos_neg = {}
    max_density = -1

    data_range = {'CC': (-1, 1), 'MR': (0, 1)}

    for label in [0, 1]:
        df = feature_df.loc[:, ['Class', rna_feature, pro_feature]]
        df = df.dropna()
        cur_df = df.loc[df['Class'] == label, [rna_feature, pro_feature]]
        hist_density, _, _ = np.histogram2d(cur_df[rna_feature].values,
                                cur_df[pro_feature].values, bins=n_bins,
                                range=np.array([data_range[feature_type],
                                                data_range[feature_type]]),
                                density=True)
        max_density = max(max_density, np.max(hist_density))

    for label in [0, 1]:
        df = feature_df.loc[:, ['Class', rna_feature, pro_feature]]
        df = df.dropna()
        cur_df = df.loc[df['Class'] == label, [rna_feature, pro_feature]]
        cnt[label] = cur_df.shape[0]
        hist, _, _ = np.histogram2d(cur_df[rna_feature].values,
                                    cur_df[pro_feature].values, bins=n_bins,
                                    range=np.array([data_range[feature_type],
                                                data_range[feature_type]]))
        cnt_pos_neg[label] = hist
        hh = ax.hist2d(cur_df[rna_feature].values,
                        cur_df[pro_feature].values,
                        bins=n_bins,
                        range=np.array([data_range[feature_type],
                            data_range[feature_type]]),
                        vmin=0,
                        vmax=max_density,
                        density=True)

    llr_vals = ((cnt_pos_neg[1] + 1)/(cnt_pos_neg[0] + cnt[0]/cnt[1]))/(cnt[1]/cnt[0])
    cmap = plt.cm.RdBu_r
    fig = ax.pcolormesh(hh[1], hh[2], np.transpose(np.log(llr_vals)), vmin=-4, vmax=4,
                        cmap=cmap)
    ax.text(0.02, 0.01, pair_name,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='gray', fontsize=24, fontweight='bold')
    ax.set_xlabel(f'{label_mapping[data_types[0]]}\n{feature_label_mapping[feature_type]}', fontsize=16)
    ax.set_ylabel(f'{label_mapping[data_types[1]]}\n{feature_label_mapping[feature_type]}', fontsize=16)
    if feature_type == 'CC':
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    else:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    return fig


def plot_pair_llr(feature_df: pd.DataFrame, output_dir: Path, rp_pairs: List[Dict[str, str]]):
    """
    Plot the heatmap of LLR for each pair of RNA and protein.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The input data frame that contains the features.
    output_dir : Path
        The output directory where the plots will be saved.
    rp_pairs : List[Dict[str, str]]
        A list of dictionaries that contain information about each RNA-protein pair, including the name and the RNA/protein features.

    Returns
    -------
    A list of file names of the plots.

    """
    n_bins = 20
    # feature_type = ['CC', 'MR']
    feature_type = ['CC']
    file_names = []
    # plot for each rna-protein pair using CC or MR as feature
    for rp_pair in rp_pairs:
        for ft in feature_type:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10),
                                gridspec_kw={'width_ratios': [4, 1],
                                            'height_ratios': [1, 4]})
            rna_feature = rp_pair['rna'] + '_' + ft
            pro_feature = rp_pair['protein'] + '_' + ft
            plot_1d_llr(ax[0,0], feature_df, rna_feature, ft, 'RNA', n_bins)
            ax[0, 0].xaxis.set_ticks_position('none')
            ax[0, 0].set_xticklabels([])

            ax[0,1].axis('off')

            heatmap2d = plot_2d_llr(ax[1,0], feature_df, ft, rp_pair['name'],
                rna_feature, pro_feature, n_bins)

            plot_1d_llr(ax[1, 1], feature_df, pro_feature, ft, 'PRO', n_bins)
            ax[1, 1].yaxis.set_ticks_position('none')
            ax[1, 1].set_yticklabels([])

            # add colorbar to the right of the plot
            cax = fig.add_axes([1.05, 0.25, 0.03, 0.5])
            fig.colorbar(heatmap2d, cax=cax)

            plt.tight_layout()
            plt.box(on=None)
            file_name = f"{rp_pair['name']}_rna_pro_{ft}_llr.pdf"
            file_names.append(file_name)
            plt.savefig(output_dir / file_name, bbox_inches='tight')

    return file_names


def plot_llr_compare_networks(validaton_results, cutoff, output_file: Path):
    """
    Plot a scatter plot to compare log likelihood ratio of different networks.

    Parameters:
    ----------
    validaton_results : dict
        A dictionary contains validation results
    cutoff : float
        Cutoff value to exclude log likelihood ratio values lower than this cutoff.
    output_file : Path
        Path to save the output plot file.

    Returns:
    -------
    n_edge:  dict
        a dictionary that contains the number of edges in the network that satisfies the cutoff
    """
    all_networks = []
    n_edge = {}

    for ft in validaton_results:
        llr_res = pd.read_csv(validaton_results[ft]['llr_res_path'], sep='\t')
        # sort the llr_res dataframe by the llr value
        llr_res = llr_res.sort_values(by=['llr'], ascending=False)

        # if the smallest llr value is larger than the cutoff, then we don't need to plot
        if llr_res['llr'].iloc[0] < np.log(cutoff):
            print('The largest llr value is smaller than the cutoff, no plot will be generated.')
            return

        # select the rows that have llr value larger than the cutoff
        llr_res = llr_res[llr_res['llr'] >= np.log(cutoff)]
        # the last row correspond the network we want
        funmap = llr_res.iloc[-1]
        # this is the number of edges in the network that satisfies the cutoff
        n_edge[ft] = int(funmap['k'])

        all_networks.extend([
            (f'FunMap_{ft}', 'FunMap', int(funmap['n']), int(funmap['k']),
                funmap['llr'], np.exp(funmap['llr']))
        ])

    # these are pre-computed values
    all_networks.extend(
    [
        # name, type, n, e, llr, lr
        ('HuRI', 'HI', 8272, 52548, 2.3139014130648827, 10.11),
        ('HI-union', 'HI', 9094, 64006, 2.298975841813893, 9.96),
        ('ProHD', 'ProHD', 2680, 61580, 4.039348296, 56.78),
        #  this is combined_score_700
        ('STRING', 'STRING', 16351, 240314,  5.229377563059293, 186.676572955849),
        ('BioGRID', 'BioGRID', 17259, 654490, 2.6524642147396182, 14.18896024552041),
        ('BioPlex', 'BioPlex', 13854, 154428, 3.3329858940888375, 28.021887299660047)
    ]
    )
    print(all_networks)

    cols = ['name', 'group', 'n', 'e', 'llr', 'lr']
    network_data = pd.DataFrame(all_networks, columns=cols)
    x = np.array(network_data['n'])
    y = np.array(network_data['llr'])
    e = np.array(network_data['e'])
    z = network_data['name']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axisbelow(True)
    ax.xaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax.yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax.get_ygridlines()[4].set_color('salmon')
    ax.get_yticklabels()[4].set_color('red')
    ax2 = ax.twinx()

    # we have 6 groups, so we need 6 colors
    mycmap = matplotlib.colors.ListedColormap(['#de2d26', '#8B6CAF', '#0D95D0',
                                            '#69A953', '#F1C36B', '#DC6C43'])
    # group 0 is FunMap, group 1 is HI, group 2 is ProHD, group 3 is STRING,
    # group 4 is BioGRID, group 5 is BioPlex
    # the length of gro
    color_group = [0] * len(validaton_results) + [1] * 2 + [2] * 1 + [3] * 1 + [4] * 1 + [5] * 1
    scatter = ax.scatter(x, y, c=color_group, cmap=mycmap,
                        s=e/1000)
    ax.set_ylim(2, 6)
    ax2.set_ylim(np.exp(2.0), np.exp(6))
    ax.set_xlabel('number of genes')
    ax.set_yticks([2.0, 2.5, 3, 3.5, np.log(cutoff), 4, 4.5, 5, 5.5, 6])
    ax.set_ylabel('log likelihood ratio')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.set_ylabel('likelihood ratio')
    ax2.set_yscale('log', base=np.e)
    ax2.set_yticks([np.exp(2), np.exp(2.5), np.exp(3), np.exp(3.5),
                    cutoff, np.exp(4), np.exp(4.5), np.exp(5),
                    np.exp(5.5), np.exp(6)])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.get_yticklabels()[4].set_color('red')
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    handles1, labels1 = scatter.legend_elements(prop='sizes', num=4, alpha=1,
                                                fmt='{x:.0f} K')
    legend1 = ax.legend(handles1, labels1,
                        loc='upper left', labelspacing=1.8, borderpad=1.0,
                        title='number of gene\npairs', frameon=True)

    ax.add_artist(legend1)
    leg = ax.get_legend()
    for i in range(len(leg.legendHandles)):
        leg.legendHandles[i].set_color('gray')

    ax.xaxis.set_major_formatter(edge_number)
    print(z)
    for i, txt in enumerate(z):
        if txt == 'STRING':
            ax.annotate(txt, (x[i]-500, y[i]+0.18), color='gray', fontsize=10)
        elif txt == 'HI-union':
            ax.annotate(txt, (x[i]-100, y[i]+0.1), color='gray', fontsize=10)
        elif txt == 'BioGRID':
            ax.annotate(txt, (x[i]-800, y[i]-0.25), color='gray', fontsize=10)
        else:
            ax.annotate(txt, (x[i]-100, y[i]-0.2), color='gray', fontsize=10)

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    return n_edge


def plot_overlap_venn(network_name, overlap, node_or_edge, color, output_dir):
    """
    Plot the Venn diagrams for the overlap between different datasets.

    Parameters
    ----------
    network_name : str
        The name of the network to plot the overlap for.
    overlap : dict
        A dictionary containing the overlap between the datasets.
        The keys are the names of the datasets, and the values are the sets
        representing the overlap.
    node_or_edge : str
        A string indicating whether to plot the overlap of nodes or edges.
        Must be one of 'node' or 'edge'.
    color : str
        The color to use for the FunMap dataset in the Venn diagrams.
    output_dir : path-like
        The directory to save the output figure in.

    Returns
    -------
    file_name : str
        The name of the file that the figure was saved as.

    """
    data = []
    for nw in overlap:
        data.append(overlap[nw])
    max_area = max(map(sum, data))

    def set_venn_scale(ax, true_area, reference_area=max_area):
        s = np.sqrt(float(reference_area)/true_area)
        ax.set_xlim(-s, s)
        ax.set_ylim(-s, s)

    all_axes = []

    n_plot = len(overlap)
    fig, ax = plt.subplots(1, n_plot, figsize=(5*n_plot, 5))

    for i, nw in enumerate(overlap):
        cur_ax = ax[i]
        all_axes.append(cur_ax)
        labels = ('FunMap', nw)
        out = venn2(overlap[nw],
                            set_labels=labels, alpha=1.0,
                            ax=cur_ax, set_colors=[color, 'white'])
        venn2_circles(overlap[nw], ax=cur_ax, linestyle='solid',
                    color='gray',
                    linewidth=1)
        if out.set_labels:
            for text in out.set_labels:
                text.set_fontsize(12)

        for text in out.subset_labels:
            text.set_fontsize(10)

    # add title to the figure
    name = 'genes' if node_or_edge == 'node' else 'edges'
    fig.suptitle(f'Overlap of {name} ({network_name})', fontsize=16)

    for a, d in zip(flatten(ax), data):
        set_venn_scale(a, sum(d)*1.5)

    file_name = f'{network_name}_overlap_{node_or_edge}.pdf'
    fig.savefig(output_dir / file_name, bbox_inches='tight')
    return file_name


def plot_network_stats(network_info, feature_type, output_dir):
    """
    Plot the network statistics for a list of networks.

    This function takes in network information and an output directory, and
    creates plots showing the degree distribution, average clustering coefficient,
    network density, and average shortest path length.

    Parameters
    ----------
    network_info : pandas.DataFrame
        A DataFrame containing information about the networks to be plotted.
    feature_type : str
        The type of feature used to create the networks, e.g 'ex', 'ei'
    output_dir : str
        The directory where the plots will be saved.

    Returns
    -------
    Name of the file that the figure was saved as.
    """

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    density = {}
    average_shortest_path = {}
    # these are pre-calculated since they take a long time to compute and
    # the network is fixed
    average_shortest_path ={
        'BioGRID': 2.74,
        'BioPlex': 3.60,
        'HI-union': 3.70,
        'STRING': 3.95
    }
    # if you want to recompute the average shortest path length,
    # add the network name to this list
    network_list = ['FunMap']
    for n in network_list:
        network_el = network_info.loc[network_info['name'] == n, 'el'].values[0]
        cur_network = nx.from_pandas_edgelist(network_el, source=0, target=1)
        cur_density = nx.density(cur_network)
        density[n] = cur_density
        largest_cc = max(nx.connected_components(cur_network), key=len)
        cur_cc = cur_network.subgraph(largest_cc).copy()
        print(n)
        cur_average_shortest_path = nx.average_shortest_path_length(cur_cc)
        average_shortest_path[n] = cur_average_shortest_path
        print(cur_average_shortest_path)
        cur_degrees = [val for (_, val) in cur_network.degree()]
        if n == 'FunMap': # only fit for FunMap
            fit = powerlaw.Fit(cur_degrees, discrete=True, xmax=250, estimate_discrete=False)
            powerlaw.plot_pdf(cur_degrees, linear_bins=True, linestyle='None', marker='o',
                        markerfacecolor='None', color='#de2d26',
                        linewidth=3, ax=ax[0])
            # not plotting the power law fit
            # fit.power_law.plot_pdf(linestyle='--',color='black', ax=ax[0])

    # all the networks in network_info minus FunMap
    other_networks = list(set(network_info['name'].tolist()) - set(['FunMap']))
    for n in other_networks:
        network_el = network_info.loc[network_info['name'] == n, 'el'].values[0]
        cur_network = nx.from_pandas_edgelist(network_el, source=0, target=1)
        cur_density = nx.density(cur_network)
        density[n] = cur_density

    ax[0].set_xlabel('degree')
    ax[0].set_ylabel('p(x)')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax[0].set_axisbelow(True)

    # global average clustering coefficient
    # these are pre-calculated since they take a long time to compute and
    # the network is fixed
    avg_cc = {
            'BioGRID': 0.125,
            'BioPlex': 0.103,
            'HI-union': 0.06,
            'STRING': 0.335
        }
    for n in network_list:
        network_el = network_info.loc[network_info['name'] == n, 'el'].values[0]
        cur_network = nx.from_pandas_edgelist(network_el, source=0, target=1)
        cur_cc = nx.average_clustering(cur_network)
        avg_cc[n] = cur_cc
        print(n, cur_cc)

    network_list = network_info['name'].tolist()
    ax[1].bar(network_list, [avg_cc[i] for i in network_list], width=0.5, align='center',
            color='#E4C89A')
    ax[1].spines['left'].set_position(('outward', 8))
    ax[1].spines['bottom'].set_position(('outward', 5))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_ylabel('Average clustering coefficient')
    ticks_loc = ax[1].get_xticks()
    ax[1].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax[1].set_xticklabels(network_list, rotation=45, ha='right')
    ax[1].yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax[1].set_axisbelow(True)

    ax[2].bar(network_list, [density[i] for i in network_list], width=0.5, align='center',
            color = '#D8B2C6'
            )
    ax[2].spines['left'].set_position(('outward', 8))
    ax[2].spines['bottom'].set_position(('outward', 5))
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].set_ylabel('Density')
    ticks_loc = ax[2].get_xticks()
    ax[2].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax[2].set_xticklabels(network_list, rotation=45, ha='right')
    ax[2].yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax[2].set_axisbelow(True)

    ax[3].bar(network_list, [average_shortest_path[i] for i in network_list], width=0.5, align='center',
            color = '#B6D8A6'
            )
    ax[3].spines['left'].set_position(('outward', 8))
    ax[3].spines['bottom'].set_position(('outward', 5))
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].set_ylabel('Average shortest path length')
    ticks_loc = ax[3].get_xticks()
    ax[3].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax[3].set_xticklabels(network_list, rotation=45, ha='right')
    ax[3].yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax[3].set_axisbelow(True)

    fig.suptitle(f'Network properties of Funmap ({feature_type})', fontsize=16)
    file_name = f'Funmap_{feature_type}_network_properties.pdf'
    fig.savefig(output_dir / file_name, bbox_inches='tight')
    return file_name


def merge_and_delete(fig_dir, file_list, output_file):
    """
    Merge multiple PDF files into one and delete the original files.

    Parameters
    ----------
    fig_dir : str or Path
        The directory where the PDF files are located.
    file_list : list of str
        The list of file names to be merged.
    output_file : str or Path
        The name of the output file.

    Returns
    -------
    None

    """
    pdf_writer = PyPDF2.PdfWriter()

    total_page_num = 0
    for file in file_list:
        pdf_reader = PyPDF2.PdfReader(fig_dir / file)
        cur_page_num = len(pdf_reader.pages)
        for page in range(cur_page_num):
            pdf_writer.add_page(pdf_reader.pages[page])
        pdf_writer.add_outline_item(os.path.splitext(file)[0], total_page_num)
        total_page_num = total_page_num + cur_page_num

    with open(fig_dir / output_file, 'wb') as fh:
        pdf_writer.write(fh)
        print('figures have been merged.')

    for filename in file_list:
        try:
            os.remove(fig_dir / filename)
        except:
            print(f'{filename} could not be deleted.')
