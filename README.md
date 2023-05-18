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
funmap qc -c test/test_config.yaml -d test/aml_test.gz -o output
```

User needs to prepare configuration file and an input data file. The configuration file is a yaml file that specifies the parameters for the experiment. The input data file is a tar gzipped file that contains the data for the experiment. A sample configuration file and a sample input data file can be found in the [`test`](https://github.com/bzhanglab/funmap/tree/main/tests) directory.

### Run the experiment

To run the experiment, use the following command

```sh
funmap run -c test/test_config.yaml -d test/aml_test.gz -o output
```

The run time of the experiment depends on the size of the input data file. The above command takes about 10 minutes to run on a standard computer.

### Hardware requirements
`funmap` package requires only a standard computer with enough RAM to support the in-memory operations.


### Output
The main output contains the predicted edge list (funmap). This network can then used for downstream analysis.

