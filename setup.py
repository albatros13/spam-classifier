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
    url='https://github.com/albatros13/spam-classifier',
    author='Natallia Kokash',
    author_email='natallia.kokash@gmail.com',
    keywords='machine learning classifier email spam filter',
    py_modules=['classifier'],
    python_requires='>=3.0.*',
    install_requires=['sklearn', 'numpy', 'joblib', 'urlextract', 'regex'],
    entry_points={
        'console_scripts': [
            'train=classifier:train_and_save',
            'predict=classifier:load_and_predict',
        ],
    },
    test_suite='test',
    project_urls={
        'Bug Reports': 'https://github.com/albatros13/spam-classifier/issues',
        'Source': 'https://github.com/albatros13/spam-classifier/',
    },
)