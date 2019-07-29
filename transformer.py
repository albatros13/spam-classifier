import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import urlextract
url_extractor = urlextract.URLExtract()


# Replace URLs and numbers
class removeWeirdThings(BaseEstimator, TransformerMixin):
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


preprocess_pipeline = Pipeline([
    ('clean', removeWeirdThings()),
    ('vect',  CountVectorizer(stop_words="english")),
    # Default values: use_idf=True, smooth_idf=True, min_df=1, lowercase=True, encoding=utf-8
    ('tfidf', TfidfTransformer())
])

