The program trains a spam classifier based on a pre-defined set of spam and non-spam emails.
The model file produced by the program can be used to get a probability of a text being a spam.

The module contains a helper, Jupyter file prepare_data.ipynb, to load and extract training and testing data 
from the SpamAssassin public mail corpus "http://spamassassin.apache.org/old/publiccorpus/"

For predicting whether a text is spam or not, call function `predict_spam`. 
For getting prediction probability, call function `spam_probability`.

## Input 
For spam classifier training, call `train_classifier` function; its `data_path` parameter expects a path to a folder with 
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

## Output 
The `train_classifier` returns a trained spam classifier or creates a file containing the spam classifier. 

## Assumptions 
It is assumed that the training data set fits to the memory and that all emails are already in clear text format.
Leftovers from the previous html markup may be present but are not cleaned up.

It is assumed that the email samples is more or less balanced, i.e., there is an approximately equal number of spam and non-spam emails. 
There are various ways of handling unbalanced classes, but with no prior information about the data set, we skip this step. 
Data manipulation methods can be added to `prepare_data.ipynb` or a decision tree based classifier can be used in `train_classifier` method.
 


  



