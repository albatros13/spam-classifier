The program trains a spam classifier based on a pre-defined set of spam and non-spam emails.
The model file produced by the program can be used to get a probability of a text being a spam.

The module contains a helper, Jupyter file `prepare_data.ipynb`, to load and extract training data 
from the SpamAssassin public mail corpus "http://spamassassin.apache.org/old/publiccorpus/"

## Input 
For spam classifier training, call `train_and_save` function; its `data_path` parameter expects a path to a folder with 
the following structure:
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
Other two parameters, `model_file` and `vectorizer_file` are optional file names to save the trained model and the term statistics.

All parameters are optional: if the data path is not given, 
the utility expects data to be in the current folder and the default file names for storage are used. 

For estimating probability of an email to be spam, call `load_and_predict` function passing the email's text, and 
saved model and vectorizer files as parameters. 

## Output 
The `train_and_save` produces two files with serialized model and term statistics for the training data. 
The `load_and_predict` prints the probability of the email input being spam.

## Assumptions 
It is assumed that the training data set fits to the memory and that all emails are already in clear text format (`utf-8` encoding is expected).
Leftovers from the previous html markup may be present but are not cleaned up. 

It is assumed that the email samples is more or less balanced, i.e., there is an approximately equal number of spam and non-spam emails. 
There are various ways of handling unbalanced classes, but with no prior information about the data set, we skip this step. 
Data manipulation methods can be added to `prepare_data.ipynb` or a decision tree classifier can be used in the `train_classifier` method.
 


  



