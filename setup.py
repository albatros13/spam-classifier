from setuptools import setup
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spam-classifier',
    version='1.0.0',
    description='A machine learning exercise that trains a spam filter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pypa/sampleproject',
    author='Natallia Kokash',
    author_email='natallia.kokash@gmail.com',
    keywords='machine learning classifier email spam filter',
    py_modules=["main"],
    python_requires='>=3.0.*',
    install_requires=['sklearn','numpy','joblib','urlextract','re'],
    entry_points={
        'console_scripts': [
            'train_and_save=main:train_and_save',
            'load_and_predict=main:load_and_predict',
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
        'Source': 'https://github.com/pypa/sampleproject/',
    },
)