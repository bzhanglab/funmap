#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'pyyaml>=6.0',
    'xgboost>=1.7.4',
    'numpy>=1.24.2',
    'scipy>=1.10.1',
    'pyarrow>=11.0.0',
    'pandas>=1.5.3',
    'joblib>=1.2.0',
    'matplotlib>=3.7.0',
    'seaborn==0.11.2',
    'scikit-learn>=1.2.1',
    'imbalanced-learn>=0.10.1',
    'tqdm>=4.64.1',
    'PyPDF2>=3.0.1',
    'matplotlib_venn>=0.11.7',
    'networkx>=3.0',
    'powerlaw>=1.5',
    'click>=8.0.1',
    'h5py>=3.4.0',
    'tables>=3.6.1',
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
    version='0.1.13',
    zip_safe=False,
)
