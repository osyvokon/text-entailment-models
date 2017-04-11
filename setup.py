from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='snli_models',
    description='Collection of neural models for text entailnment task',
    long_description=long_description,

    url='https://github.com/asivokon/snli-neural-models',
    license='MIT',

    keywords='snli text entailment nural attention lstm nlp',

    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=[
        'pytest',
        'keras',
        'recurrentshop',
    ],

    extras_require={
        'test': ['pytest'],
    },

)
