import numpy as np
import os
import joblib
# Other classifiers can be used, e.g., GaussianNB, logistic, svm
from sklearn.naive_bayes import MultinomialNB
from transformer import preprocess_pipeline


DATA_DIR = os.path.join("training_data")
SPAM = "spam"
NON_SPAM = "non_spam"


def train_classifier(data_path=DATA_DIR):
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

    X_transformed = preprocess_pipeline.fit_transform(X)

    # Other classifiers can be used instead, e.g. logistic, svm, GaussianNB,...
    clf = MultinomialNB()
    clf.fit(X_transformed, y)
    return clf


def spam_probability(clf, email):
    email_transformed = preprocess_pipeline.transform([email])
    p = clf.predict_proba(email_transformed)
    return p


def predict_spam(clf, email):
    email_transformed = preprocess_pipeline.transform([email])
    p = clf.predict(email_transformed)
    return NON_SPAM if p[0] == 0 else SPAM

# Save the classifier
def save_classifier(clf, model_file="saved_model.pk1"):
    joblib.dump(clf, model_file)


# Load the classifier
def load_classifier(model_file="saved_model.pk1"):
    return joblib.load(model_file)


# TODO save vectorizer vocabulary
# joblib.dump(vectorizer.vocabulary_, dictionary_file_path)
# TODO read saved vocubulary
# vocabulary_to_load =joblib.load(dictionary_file_path)
# loaded_vectorizer = CountVectorizer(vocabulary=vocabulary_to_load)

