import numpy as np
import os
import joblib
import traceback
import logging
import sys
import regex as re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import urlextract
url_extractor = urlextract.URLExtract()

DATA_DIR = os.path.join("training_data")
SPAM = "spam"
NON_SPAM = "non_spam"
DEFAULT_MODEL_FILE = "saved_model.pk1"


# Replace non-informative things in the email text such as URLs and numbers
class ReplaceWeirdThings(BaseEstimator, TransformerMixin):
    def __init__(self, replace_urls=True, replace_numbers=True):
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed=[]
        for text in X:
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            X_transformed.append(text)
        return X_transformed


pipeline = Pipeline([
    ('clean', ReplaceWeirdThings()),
    ('vect', TfidfVectorizer(stop_words="english")),
    ('clf', MultinomialNB())
])


# Load emails from spam and non-spam sub-directories of a given directory
def load_data(data_path=""):
    data_dir = os.path.join(data_path, DATA_DIR)
    if not os.path.isdir(data_dir):
        print("Training data directory not found!")
        return None, None
    spam_dir = os.path.join(data_dir, SPAM)
    non_spam_dir = os.path.join(data_dir, NON_SPAM)
    if not os.path.isdir(spam_dir):
        print("Spam directory not found!")
        return None, None
    if not os.path.isdir(non_spam_dir):
        print("Non-spam directory not found!")
        return None, None

    print("Loading data...")

    spam_file_names = [name for name in os.listdir(spam_dir)]
    non_spam_file_names = [name for name in os.listdir(non_spam_dir)]

    # Load email text from the predefined folders
    def load_email_text(is_spam, filename):
        directory = spam_dir if is_spam else non_spam_dir
        with open(os.path.join(directory, filename), "rb") as f:
            return f.read().decode('utf-8')

    spam_emails = [load_email_text(is_spam=True, filename=name) for name in spam_file_names]
    non_spam_emails = [load_email_text(is_spam=False, filename=name) for name in non_spam_file_names]

    X = np.array(non_spam_emails + spam_emails)
    y = np.array([0] * len(non_spam_emails) + [1] * len(spam_emails))  # 0-ham & 1-spam
    return X,y


# save_classifier
def save_classifier(clf, model_file=DEFAULT_MODEL_FILE):
    joblib.dump(clf, model_file)
    print('The classifier has been saved in a file: {}'.format(model_file))


# Load classifier
def load_classifier(model_file=DEFAULT_MODEL_FILE):
    if not os.path.isfile(model_file):
        print("Model file not found!")
        return None
    return joblib.load(model_file)


# Create, train, and save classifier and the vocabulary for given data in two files
def train_and_save():
    try:
        data_path = ""
        model_file = DEFAULT_MODEL_FILE
        args = sys.argv[1:]
        if args:
            if len(args) > 0:
                data_path = args[0]
            if len(args) > 1:
                model_file = args[1]

        # print("Proceeding with arguments '{}', '{}': ".format(data_path, model_file))
        X, y = load_data(data_path)
        if X is not None:
            clf = pipeline.fit(X, y)
            print('The classifier has been trained!')
            save_classifier(clf, model_file)
    except:
        print("Something unexpected happened! See logs for more information.")
        logging.error(traceback.format_exc())


# Load the classifier and the vectorizer and estimate probability of spam for a given email
def load_and_predict():
    try:
        model_file = DEFAULT_MODEL_FILE
        args = sys.argv[1:]
        if len(args) == 0:
            print("The command expects at least one argument - an email text!")
            return
        email = args[0]
        if len(args) > 1:
            model_file = args[1]
        print("Proceeding with arguments '{}', '{}': ".format(email, model_file))
        clf = load_classifier(model_file)
        if clf is not None:
            p = clf.predict_proba([email])
            p = round(p[0][1], 2)
            print("The predicted probability of the email being spam is:", p)
    except:
        print("Something unexpected happened! See logs for more information.")
        logging.error(traceback.format_exc())
