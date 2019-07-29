import numpy as np
import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from transformer import preprocess_pipeline
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join("training_data")
SPAM = "spam"
NON_SPAM = "non_spam"
DEFAULT_MODEL_FILE = "saved_model.pk1"
DEFAULT_PIPELINE_FILE = "saved_pipeline.pk1"


# Load emails from spam and non-spam sub-directories of a given directory
def load_data(data_path=DATA_DIR):
    spam_dir = os.path.join(data_path, SPAM)
    non_spam_dir = os.path.join(data_path, NON_SPAM)

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
    X_transformed = preprocess_pipeline.fit_transform(X)
    # clf = MultinomialNB() # other classifiers can be used, e.g., LogisticRegression, svm, GaussianNB
    clf=LogisticRegression(solver="liblinear", random_state=42)
    clf.fit(X_transformed, y)
    return clf


# Get probability that email is spam
def spam_probability(clf, email):
    email_transformed = preprocess_pipeline.transform([email])
    p = clf.predict_proba(email_transformed)
    return p[0][1]


# Save the classifier
def save_classifier(clf, model_file=DEFAULT_MODEL_FILE, pipeline_file=DEFAULT_PIPELINE_FILE):
    joblib.dump(clf, model_file)
    joblib.dump(preprocess_pipeline, pipeline_file)


# Load the classifier
def load_classifier(model_file="saved_model.pk1", pipeline_file=DEFAULT_PIPELINE_FILE):
    global preprocess_pipeline
    preprocess_pipeline = joblib.load(pipeline_file)
    return joblib.load(model_file)


