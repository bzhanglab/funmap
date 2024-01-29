#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'pyyaml==6.0.1',
    'xgboost==2.0.0',
    'numpy==1.24.4',
    'scipy==1.10.1',
    'pyarrow==13.0.0',
    'pandas==2.0.3',
    'joblib==1.3.2',
    'matplotlib==3.7.3',
    'seaborn==0.13.0',
    'scikit-learn==1.3.2',
    'imbalanced-learn==0.11.0',
    'tqdm==4.66.1',
    'PyPDF2==3.0.1',
    'matplotlib_venn==0.11.9',
    'networkx==3.1',
    'powerlaw==1.5',
    'click==8.1.7',
    'h5py==3.10.0',
    'tables==3.8.0'
    ]

test_requirements = [ ]

setup(
    author="Zhiao Shi",
    author_email='zhiao.shi@gmail.com',
    python_requires='>=3.7',
    description="generate gene co-function networks using omics data",
    entry_points={
        'console_scripts': [
            'funmap=funmap.cli:cli',
        ],
    },
    install_requires=requirements,
    license='MIT license',
    include_package_data=True,
    keywords=['funmap', 'bioinformatics', 'biological-network'],
    name='funmap',
    packages=find_packages(include=['funmap', 'funmap.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/bzhanglab/funmap',
    version='0.1.19',
    zip_safe=False,
)
