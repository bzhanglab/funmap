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
funmap -h
usage: funmap [-h] -c CONFIG_FILE -d DATA_FILE [-o OUTPUT_DIR] [--version]

command line arguments.

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        path to experiment configuration yaml file
  -d DATA_FILE, --data-file DATA_FILE
                        path to tar gzipped data file
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        path to output directory
  --version             show program's version number and exit
```

User needs to prepare configuration file and an input data file. The configuration file is a yaml file that specifies the parameters for the experiment. The input data file is a tar gzipped file that contains the data for the experiment. A sample configuration file and a sample input data file can be found in the `test` directory.

To run the experiment, use the following command

```sh
funmap -c test/test_config.yaml -d test/aml_test.gz -o output
```

The run time of the experiment depends on the size of the input data file. The above command takes about 10 minutes to run on a standard computer.

### Hardware requirements
`funmap` package requires only a standard computer with enough RAM to support the in-memory operations.


### Output
The main output contains the predicted edge list (funmap). This network can then used for downstream analysis.

