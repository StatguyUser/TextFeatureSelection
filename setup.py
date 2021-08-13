from setuptools import setup
import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name='TextFeatureSelection',
    version='0.0.15',
    description='Python library for feature selection for text features. It has filter method, genetic algorithm and TextFeatureSelectionEnsemble for improving text classification models. Helps improve your machine learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    author='StatguyUser',
    url='https://github.com/StatguyUser/TextFeatureSelection',
    install_requires=['numpy','pandas','scikit-learn','xgboost','nltk','EvolutionaryFS'],
    download_url='https://github.com/StatguyUser/TextFeatureSelection.git',
    py_modules=["TextFeatureSelection"],
    package_dir={'':'src'},
)
