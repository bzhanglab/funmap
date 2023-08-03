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

Before running the experiment, user can check the quality of the input data using the following command

```sh
funmap qc -c test_config.yaml -d aml_test.tgz -o output
```

User needs to prepare configuration file and an input data file. The configuration file is a YAML file that specifies the parameters for the experiment. The input data file is a tar gzipped file that contains the data for the experiment. A sample configuration file and a sample input data file can be found in the [`test`](https://github.com/bzhanglab/funmap/tree/main/tests) directory.

### Run the experiment

To run the experiment, use the following command

```sh
funmap run -c test_config.yaml -d aml_test.tgz -o output
```

The run time of the experiment depends on the size of the input data file. The above command takes about 20-30 minutes to run on a standard computer using 4 threads.

#### Configuration file

| Field             | Description                                                                                       | Example Value     |
|-------------------|---------------------------------------------------------------------------------------------------|-------------------|
| seed              | Seed for random number generation.                                                                | 42                |
| cor_type          | Type of correlation, can be 'pearson' or 'spearman'.                                              | 'pearson'         |
| feature_type      | Type of features used for training a model, can be 'cc' (correlation coefficient) or 'mr' (mutual rank).                                              | 'cc'         |
| n_jobs            | Number of parallel jobs to run.                                                                   | 8                 |
| n_chunk           | Number of chunks to split the data for parallel processing.                                      | 4                 |
| start_edge_num    | The starting number for evaluating the Log-Likelihood Ratio (LLR) for each individual data set.  | 100               |
| min_sample_count  | Minimum number of valid data points required when computing correlation.                         | 15                |
| output_edgelist   | Whether to output the edge list or not.                                                          | True              |
| max_num_edges     | Maximum number of edges to consider.                                                              | 25000             |
| step_size         | Step size for the evaluation of LLR.                                                              | 1                 |
| lr_cutoff         | Likelihood Ratio (LR) cutoff value.                                                          | 10                |
| dataset_name      | Name of the dataset.                                                                              | 'aml'             |
| data_files        | List of data files with their name, type, and path.                                               | See the example   |



### Hardware requirements
`funmap` package requires only a standard computer with enough RAM to support the in-memory operations.


## Output
The output directory contains the following files and directories:

```
.
├── config.json
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


* `config.json`: the configuration file used for the experiment
* `figures`: the directory that contains the figures generated by the experiment. If QC was performed, the figures will be saved in this directory also.
* `llr_dataset.tsv`: a tsv file contains log-likelihood ratio (LLR) analysis for each individual input data set.
* `llr_results_ei_25000.tsv`: a tsv file contains LLR analysis for predictions based on the model trained with mutual rank and PPI features. The number in the file name indicates the maximum number of edges selected for LLR analysis.
* `llr_results_ex_25000.tsv`: a tsv file contains LLR analysis for predictions based on the model trained with mutual rank features.
* `networks`: the directory that contains the predicted networks. The network files are tab-separated files with three columns: gene1, gene2, and score. The score is the predicted probability of the edge between gene1 and gene2. `funmap.tsv` is the final predicted network. The edges meet the required LLR threshold.
* `saved_data`: the directory that contains the saved data used for the experiment.
* `saved_models`: the directory that contains the trained model.
* `saved_predictions`: the directory that contains the predicted probabilities for all pairs of genes.
