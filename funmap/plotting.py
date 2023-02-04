from typing import List, Union, Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from funmap.utils import get_data_dict
import matplotlib
import matplotlib.pyplot as plt


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


def plot_llr_comparison(llr_res, llr_ds, output_file='llr_comparison.pdf'):
    """
    Plot the comparison of log likelihood ratios (LLR) based on model prediction
    using all datasets and LLR results for each dataset.

    Parameters
    ----------
    llr_res : pandas DataFrame
        LLR results based on model prediction using all datasets
    llr_ds : pandas DataFrame
        LLR results for each dataset
    output_file : str/Path, optional
        Output file name or path for saving the plot, by default 'llr_comparison.pdf'

    Returns
    -------
    None

    """
    datasets = sorted(llr_ds['dataset'].unique().tolist())
    fig, ax = plt.subplots(figsize=(5, 4))

    start = -1
    for ds in datasets:
        cur_df = llr_ds[llr_ds['dataset'] == ds]
        ax.plot(cur_df['k'], cur_df['llr'], label=ds)
        if start == -1:
            start = cur_df['k'].iloc[0]
    # plot llr_res with the same start point
    llr_res = llr_res[llr_res['k'] >= start]
    ax.plot(llr_res['k'], llr_res['llr'], label='all datasets', color='black', linewidth=2)

    ax.xaxis.set_major_formatter(edge_number)
    ax.set_xlabel('number of gene pairs')
    ax.set_ylabel('log likelihood ratio')
    ax.yaxis.grid(color = 'gainsboro', linestyle = 'dotted')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.box(on=None)
    plt.savefig(output_file, bbox_inches='tight')


def explore_data(data_config: Path, min_sample_count: int,
                output_dir: Path):
    """
    Generate plots to explore and visualize data

    Parameters
    ----------
    data_config: Path
        Path to the data configuration file
    min_sample_count: int
        The minimum number of samples required to consider a dataset
    output_dir: Path
        The directory to save the output plots

    Returns
    -------
    None

    """
    data_dict = get_data_dict(data_config, min_sample_count)

    # sample wise median expression plot for each dataset
    data = []
    data_keys = []

    for ds in data_dict:
        data_df = data_dict[ds]
        fig, ax = plt.subplots(figsize=(10, 5))
        cur_data = data_df.T
        cur_data.dropna(inplace=True)
        ax.boxplot(cur_data)
        ax.set_ylabel('expression')
        if data_df.shape[0] > 100:
            ax.set_xlabel(f'sample (n={data_df.shape[0]})')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('sample')
        fig.tight_layout()
        fig.savefig(output_dir/f'{ds}_sample_box_plot.pdf')
        data_keys.append(ds)
        data.append(data_df.median(axis=1).values)

    # from IPython import embed; embed()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data)
    ax.set_xticklabels(data_keys, rotation=45)
    ax.yaxis.grid(color = 'gainsboro', linestyle = 'dotted')

    ax.set_xlabel('dataset')
    ax.set_ylabel('median expression')
    plt.box(on=None)
    fig.tight_layout()
    fig.savefig(output_dir/f'data_box_plot.pdf')

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
    fig.savefig(output_dir/f'sample_gene_count.pdf')


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
    None

    """
    n_bins = 20
    # feature_type = ['CC', 'MR']
    feature_type = ['CC']
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
            cb = fig.colorbar(heatmap2d, cax=cax)

            plt.tight_layout()
            plt.box(on=None)
            plt.savefig(output_dir/f"{rp_pair['name']}_rna_pro_{ft}_llr.pdf",
                        bbox_inches='tight')


def plot_llr_compare_networks(llr_res, cutoff, output_file: Path):
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

    all_networks = [
        ('FunMap', 'FunMap', int(funmap['n']), int(funmap['k']),
            funmap['llr'], np.exp(funmap['llr']))
    ]

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

    mycmap = matplotlib.colors.ListedColormap(['#de2d26', '#8B6CAF', '#0D95D0',
                                            '#69A953', '#F1C36B', '#DC6C43'])
    scatter = ax.scatter(x, y, c=[0, 1, 1, 2, 3, 4, 5], cmap=mycmap,
                        s=e/1000)
    ax.set_ylim(2, 6)
    ax2.set_ylim(np.exp(2.0), np.exp(6))
    ax.set_xlabel('number of genes')
    ax.set_yticks([2.0, 2.5, 3, 3.5, 3.91, 4, 4.5, 5, 5.5, 6])
    ax.set_ylabel('log likelihood ratio')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax2.set_ylabel('likelihood ratio')
    ax2.set_yscale('log', base=np.e)
    ax2.set_yticks([np.exp(2), np.exp(2.5), np.exp(3), np.exp(3.5),
                    50, np.exp(4), np.exp(4.5), np.exp(5),
                    np.exp(5.5), np.exp(6)])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.get_yticklabels()[4].set_color('red')
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)

    handles1, labels1 = scatter.legend_elements(prop='sizes', num=4, alpha=1,
                                                fmt='{x:.0f} K')
    # legend1 = ax.legend(handles1, labels1, bbox_to_anchor=(1.1, 1.0),
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
