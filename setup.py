# -*- coding:utf-8 -*-
# Narrative Operator NLP — NLP operator for NarrativeOS
# Note: the `hanlp/` package is a vendored fork of the HanLP engine and keeps
# its own version (hanlp/version.py). This project's version is independent.
from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()
__version__ = '0.1.0'

FASTTEXT = 'fasttext-wheel==0.9.2'

extras_require = {
    'amr': [
        'penman==1.2.1',
        'networkx>=2.5.1',
        'perin-parser>=0.0.12',
    ],
    'fasttext': [FASTTEXT],
    'tf': [FASTTEXT, 'tensorflow>=2.6.0,<2.14', "transformers<4.55"],
    # NarrativeOS protocol adapters
    'narrative': [
        'pydantic>=2.0',
        'fastapi>=0.100',
        'uvicorn[standard]>=0.23',
        'grpcio>=1.50',
        'grpcio-tools>=1.50',
        'mcp>=1.0',
    ],
}
extras_require['full'] = list(set(sum(extras_require.values(), [])))

setup(
    name='narrative-operator-nlp',
    version=__version__,
    description='NLP operator for NarrativeOS — protocol-first NLP analysis built on HanLP',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/narrativeos/narrative-operator-nlp',
    author='NarrativeOS',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Development Status :: 4 - Beta",
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Text Processing :: Linguistic"
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    packages=find_packages(exclude=['docs', 'tests*', 'plugins*']),
    include_package_data=True,
    install_requires=[
        'termcolor',
        'pynvml',
        'toposort==1.5',
        'transformers>=4.30,<5.0',  # HanLP 2.1.x compat: encode_plus removed in 5.x
        'sentencepiece>=0.1.91',  # Essential for tokenization_bert_japanese
        'torch>=1.6.0',
        'hanlp-common>=0.0.23',
        'hanlp-trie>=0.0.4',
        'hanlp-downloader',
        'pyyaml>=5.0',  # Required for config-driven entity extraction
    ],
    extras_require=extras_require,
    python_requires='>=3.10',
)
