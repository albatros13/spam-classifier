import numpy as np
import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from transformer import preprocess_pipeline
# from sklearn.linear_model import LogisticRegression
import traceback
import logging

DATA_DIR = os.path.join("training_data")
SPAM = "spam"
NON_SPAM = "non_spam"
DEFAULT_MODEL_FILE = "saved_model.pk1"
DEFAULT_VECTORIZER_FILE = "saved_vectorizer.pk1"


# Load emails from spam and non-spam sub-directories of a given directory
def load_data(data_path=DATA_DIR):
    if not os.path.isdir(data_path):
        print("Training data directory not found!")
        return
    spam_dir = os.path.join(data_path, SPAM)
    non_spam_dir = os.path.join(data_path, NON_SPAM)
    if not os.path.isdir(spam_dir):
        print("Spam directory not found!")
        return
    if not os.path.isdir(non_spam_dir):
        print("Non-spam directory not found!")
        return

    print("Loading data...")

    spam_file_names = [name for name in os.listdir(spam_dir)]
    non_spam_file_names = [name for name in os.listdir(non_spam_dir)]

    # Load email text from the predefined folders
    def load_email_text(is_spam, filename):
        directory = SPAM if is_spam else NON_SPAM
        with open(os.path.join(data_path, directory, filename), "rb") as f:
            return f.read().decode('utf-8')

    spam_emails = [load_email_text(is_spam=True, filename=name) for name in spam_file_names]
    non_spam_emails = [load_email_text(is_spam=False, filename=name) for name in non_spam_file_names]

    X = np.array(non_spam_emails + spam_emails)
    y = np.array([0] * len(non_spam_emails) + [1] * len(spam_emails))  # 0-ham & 1-spam
    return X,y


# Train spam filter
def train_classifier(X,y):
    print("Training classifier...")
    X_transformed = preprocess_pipeline.fit_transform(X)
    clf = MultinomialNB()
    # other classifiers can be used, e.g., LogisticRegression, svm, GaussianNB
    # clf = LogisticRegression(solver="liblinear", random_state=42)
    clf.fit(X_transformed, y)
    return clf


# Get probability that an email is spam
def spam_probability(clf, email):
    email_transformed = preprocess_pipeline.transform([email])
    p = clf.predict_proba(email_transformed)
    return p[0][1]


# Save classifier
def save_classifier(clf, model_file=DEFAULT_MODEL_FILE, vectorizer_file=DEFAULT_VECTORIZER_FILE):
    joblib.dump(clf, model_file)
    joblib.dump(preprocess_pipeline.named_steps["tfidf"], vectorizer_file)


# Load classifier
def load_classifier(model_file="saved_model.pk1", vectorizer_file=DEFAULT_VECTORIZER_FILE):
    if not os.path.isfile(model_file):
        print("Model file not found!")
        return
    if not os.path.isfile(vectorizer_file):
        print("Vectorizer file not found!")
        return
    preprocess_pipeline.named_steps["tfidf"] = joblib.load(vectorizer_file)
    return joblib.load(model_file)


# Create, train, and save classifier and the vectorizer (vocabulary for given data) in two files
def train_and_save(data_path=DATA_DIR, model_file=DEFAULT_MODEL_FILE, vectorizer_file=DEFAULT_VECTORIZER_FILE):
    try:
        X, y = load_data(data_path)
        cls = train_classifier(X, y)
        save_classifier(cls, model_file, vectorizer_file)
        print('The classifier has been trained and saved in a file: {}'.format(model_file))
        print('Term weights for the training data have been saved in a file: {}'.format(vectorizer_file))
    except:
        print("Something unexpected happened! See logs for more information.")
        logging.error(traceback.format_exc())


# Load the classifier and the vectorizer and estimate probability of spam for a given email
def load_and_predict(email, model_file=DEFAULT_MODEL_FILE, vectorizer_file=DEFAULT_VECTORIZER_FILE):
    try:
        cls = load_classifier(model_file, vectorizer_file)
        p = spam_probability(cls, email)
        print("The predicted probability of the email being spam is:", p)
        return p
    except:
        print("Something unexpected happened! See logs for more information.")
        logging.error(traceback.format_exc())

