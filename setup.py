#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'scipy',
    'pandas',
    'joblib',
    'scikit-learn',
    'imblearn',
    'tqdm'
    ]

test_requirements = [ ]

setup(
    author="Zhiao Shi",
    author_email='zhiao.shi@gmail.com',
    python_requires='>=3.7',
    description="generate gene co-function networks using omics data",
    entry_points={
        'console_scripts': [
            'funmap=funmap.cli:main',
        ],
    },
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    package_data={
        'funmap': ['data/*.tsv', 'data/*.txt']
    },
    keywords=['funmap', 'bioinformatics', 'biological-network'],
    name='funmap',
    packages=find_packages(include=['funmap', 'funmap.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/bzhanglab/funmap',
    version='0.1.3',
    zip_safe=False,
)
