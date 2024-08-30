# funmap
`funmap` integrates multiple omics data sets (such as proteomics and RNASeq) to construct a functional network using supervised machine learning (xgboost).

## Installation

### Dependencies

`funmap` requires the following:

* [python](https://www.python.org/) (>= 3.7)
* [numpy](https://numpy.org/)  (>= 1.17)
* [scipy](https://docs.scipy.org/doc/scipy/reference/) (>= 1.4.0)
* [scikit-learn](https://scikit-learn.org/stable/) (>= 0.22)
* [joblib](https://joblib.readthedocs.io/en/latest/) (>= 0.17.0)

### User installation

The easiest way to install `funmap` is using `pip`

```sh
pip install funmap
```

To upgrade to a newer release use the `--upgrade` flag

```sh
pip install --upgrade funmap
```

## How to run

```sh
usage: funmap [-h] [--version] {qc,run} ...

funmap command line interface

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

Commands:
  {qc,run}
    qc        check the data quality
    run       run funmap
```

### Data quality check

Before running the experiment, user can check the quality of the input data using the following command in the project directory

```sh
funmap qc -c test_config.yml
```

User needs to prepare configuration file and an input data file. The configuration file is a YAML file that specifies the parameters for the experiment. The input data file is a tar gzipped file that contains the data for the experiment. A sample configuration file and a sample input data file can be found in the [`test`](https://github.com/bzhanglab/funmap/tree/main/tests) directory.

### Run the experiment

To run the experiment, use the following command in the project directory

```sh
funmap run -c test_config.yml
```

The run time of the experiment depends on the size of the input data file. The above command takes about 20-30 minutes to run on a standard computer using 4 threads.

#### Configuration file

| Item                     | Description                                                                                          | Example Value        |
|--------------------------|------------------------------------------------------------------------------------------------------|----------------------|
| `task`                   | For now always set to `protein_func`                                                                 | `protein_func`       |
| `name`                   | Unique identifier for the experiment.                                                                | `experiment_name`    |
| `seed`                   | Random seed for reproducibility.                                                                     | `42`                 |
| `results_dir`            | Directory where output results will be stored.                                                       | `results`            |
| `filter_noncoding_genes` | Setting to exclude non-coding genes from analysis (`True` or `False`).                               | `True`               |
| `cor_type`               | Type of correlation, can be `pearson` or `spearman`.                                                 | `pearson`            |
| `feature_type`           | Type of features to be used in the analysis. `mr` (mutual rank) or `cc` (correlation coefficient).   | `mr`                 |
| `n_jobs`                 | Number of parallel jobs or threads to use for processing.                                            | `40`                 |
| `min_sample_count`       | Minimum number of samples required for calculating correlation.                                      | `15`                 |
| `start_edge_num`         | Starting number of edges for calculating LR.                                                         | `1000`               |
| `max_num_edges`          | Maximum number of edges to consider in network analysis.                                             | `250000`             |
| `step_size`              | Step size for incrementing edges in LR analysis.                                                     | `100`                |
| `lr_cutoff`              | Cutoff threshold for LR (likelihood ratio) in LR analysis.                                           | `50`                 |
| `data_path`              | Name of the compressed dataset file containing all necessary data files.                             | `Dataset_Name.tgz`   |

Note: `data_path` should be the name of the tar gzipped file that contains all the data files. It should be placed in the same directory as the configuration file.

```
project_directory/
│
├── config.yml
│
└── dataset_name/
    ├── protein_data_1.tsv
    ├── protein_data_2.tsv
    ├── rna_data_1.tsv
    └── rna_data_2.tsv
```

When in the project_directory, run the following command to compress the dataset:
```sh
tar -czvf Dataset_Name.tgz dataset_name/
```

**`data_files` Section**

The `data_files` section specifies the list of data files used in the analysis. Each entry includes a unique name, the type of data (protein or RNA), and the path to the data file.

| Field        | Description                                | Example Value                        |
|--------------|--------------------------------------------|---------------------------------------|
| `name`       | Unique identifier for the data file.       | `'protein_data_file_1'`              |
| `type`       | Type of data (`'protein'` or `'rna'`).     | `'protein'`                          |
| `path`       | the data file name within the dataset.     | `'protein_data_1.tsv'`               |

Example Entries for `data_files`:

```yaml
data_files:
  - name: 'protein_data_file_1'
    type: 'protein'
    path: 'protein_data_1.tsv'
  - name: 'protein_data_file_2'
    type: 'protein'
    path: 'protein_data_2.tsv'
  - name: 'rna_data_file_1'
    type: 'rna'
    path: 'rna_data_1.tsv'
```

**`rp_pairs` Section** (Optional)

The `rp_pairs` section defines RNA-protein pairs for analysis. Each entry should include a unique identifier for the pair, along with the corresponding RNA and protein data file names from the `data_files` section.

| Field      | Description                                           | Example Value                      |
|------------|-------------------------------------------------------|-------------------------------------|
| `name`     | Unique identifier for the RNA-protein pair.           | `'rna_protein_pair_1'`             |
| `rna`      | Identifier of the RNA data file from `data_files`.    | `'rna_data_file_1'`                |
| `protein`  | Identifier of the protein data file from `data_files`.| `'protein_data_file_1'`            |

### Example Entries for `rp_pairs`:

```yaml
rp_pairs:
  - name: 'rna_protein_pair_1'
    rna: 'rna_data_file_1'
    protein: 'protein_data_file_1'
  - name: 'rna_protein_pair_2'
    rna: 'rna_data_file_2'
    protein: 'protein_data_file_2'
```



### Hardware requirements
`funmap` package requires only a standard computer with enough RAM to support the in-memory operations.


## Output
The output directory contains the following files and directories:

```
.
├── config.yml
├── figures
│   └── results.pdf
├── llr_dataset.tsv
├── llr_results_ei_25000.tsv
├── llr_results_ex_25000.tsv
├── networks
│   ├── funmap.tsv
│   ├── network_ei_25000.tsv
│   └── network_ex_25000.tsv
├── saved_data
│   ├── all_features.fth
│   ├── all_pairs.tsv.gz
│   ├── all_valid_gene.txt
│   ├── gold_standard_test_neg.pkl.gz
│   ├── gold_standard_test_pos.pkl.gz
│   └── gold_standard_train.pkl.gz
├── saved_models
│   └── model.pkl.gz
└── saved_predictions
    └── predicted_all_pairs.pkl.gz
```


* `config.yml`: the configuration file used for the experiment
* `figures`: the directory that contains the figures generated by the experiment. If QC was performed, the figures will be saved in this directory also.
* `llr_dataset.tsv`: a tsv file contains log-likelihood ratio (LLR) analysis for each individual input data set.
* `llr_results_ei_25000.tsv`: a tsv file contains LLR analysis for predictions based on the model trained with mutual rank and PPI features. The number in the file name indicates the maximum number of edges selected for LLR analysis.
* `llr_results_ex_25000.tsv`: a tsv file contains LLR analysis for predictions based on the model trained with mutual rank features.
* `networks`: the directory that contains the predicted networks. The network files are tab-separated files with three columns: gene1, gene2, and score. The score is the predicted probability of the edge between gene1 and gene2. `funmap.tsv` is the final predicted network. The edges meet the required LLR threshold.
* `saved_data`: the directory that contains the saved data used for the experiment.
* `saved_models`: the directory that contains the trained model.
* `saved_predictions`: the directory that contains the predicted probabilities for all pairs of genes.
