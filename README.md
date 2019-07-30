A console utility that trains a spam classifier based on a pre-defined set of spam and non-spam emails.
The model file produced by the utility can be used to get a probability of a text being a spam.

The tool relies on the [Scikit-learn](http://scikit-learn.org) library and uses 
[Multinomial naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) classification algorithm with 
[tf-idf](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) statistics.
 

The source code contains a helper, Jupyter file `prepare_data.ipynb`, to load and extract training data 
from the [SpamAssassin public mail corpus](http://spamassassin.apache.org/old/publiccorpus/)

The `test.py` file contains unit tests to evaluate classifier performance. 

## Installation
The utility can be packaged and installed from the source code using the following commands
```
    $ python setup.py sdist bdist_wheel
    $ pip install dist/spam-classifier-1.0.0.tar.gz
```

# Usage examples
After installation, call `$ train` to train the classifier on data in the current folder and output to the default model file.
Alternatively, provide the path to the training data and/or the output file name, e.g. `$ train "c:/" my_model.mdl`

To estimate the probability of an email being spam, call `predict` followed by the email text, e.g.,
`$ predict "Dear Winner, we wish to congratulate and inform you that your email address has won ($2,653,000 two million six hundred and fifty three thousand US Dollars)" `

## Input 
The `train` entry point invokes the `train_and_save` function; its `data_path` parameter expects a path to a folder with the following structure:
```
training_data
    spam
        email_1.txt
        email_2.txt
        ...
        email_N.txt
    non_spam
        email_1.txt
        email_2.txt
        ...
        email_M.txt
```
The second parameter, `model_file` is a file name to save the trained model.

Both parameters are optional: if the data path is not provided, 
the utility expects data to be in the current folder. 
By default, the model is stored in the file `saved_model.pk1`. 

The `predict` entry point invokes the `load_and_predict` function passing the email's text and the  
saved classifier as parameter. The latter can be skipped if the default file name was used.  

## Output 
The `train_and_save` produces a file with serialized pipeline (pre-processor, vectorizer and classification algorithm). 
The `load_and_predict` prints the probability of the email input being spam.

## Assumptions 
It is assumed that the training data set fits to the memory and that all emails are already in clear text format (`utf-8` encoding is expected).
Leftovers from the previous html markup may be present but are not cleaned up. 

It is assumed that the email samples is more or less balanced, i.e., there is an approximately equal number of spam and non-spam emails. 
There are various ways of handling unbalanced classes, but with no prior information about the data set, we skip this step. 
Data manipulation methods can be added to `prepare_data.ipynb` or a decision tree classifier can be used in the pipeline.
 


  



