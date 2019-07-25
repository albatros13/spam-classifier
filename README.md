The program trains a spam classifier based on a pre-defined set of spam and non-spam emails.
The model file produced by the program can be used to get a probability of a text being a spam.

The module contains a helper, Jupyter file prepare_data.ipynb, to load and extract training and testing data 
from the SpamAssassin public mail corpus "http://spamassassin.apache.org/old/publiccorpus/"

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

All emails must be already in clear text format. 
Leftovers from the previous html markup may be present and will not be cleaned up.

For predicting whether a text is spam or not, call function `predict_spam`. 
For getting prediction probability, call function `spam_probability`.

## Output 
The `train_classifier` returns a trained spam classifier or creates a file containing the spam classifier. 



