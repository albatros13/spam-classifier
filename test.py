import unittest
import numpy as np
from main import load_data, train_classifier, spam_probability, save_classifier, load_classifier
from transformer import preprocess_pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


class TestClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestClassifier, self).setUpClass()

        # load_data and train_test_split may be slow, to avoid calling it for each test we store the result as class variables
        X, y = load_data()
        print('Loaded {} emails and {} labels'.format(len(X), len(y)))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('Split data into {} training samples and {} testing samples'.format(len(self.X_train), len(self.X_test)))

        self.clf = train_classifier(self.X_train, self.y_train)

    # Test classifier performance
    def test_classifier(self):
        X_test_transformed = preprocess_pipeline.transform(self.X_test)
        y_pred = self.clf.predict(X_test_transformed)
        precision = 100 * precision_score(self.y_test, y_pred)
        recall = 100 * recall_score(self.y_test, y_pred)
        mean = 100 * np.mean(y_pred == self.y_test)

        # print("Precision: {:.2f}%".format(precision))
        # print("Recall: {:.2f}%".format(recall))
        # print("Mean: {:.2f}%".format(mean))

        self.assertGreaterEqual(precision, 80)
        self.assertGreaterEqual(recall, 80)
        self.assertGreaterEqual(mean, 80)

    # Test save and load, compare loaded model with the persisted one
    def test_save_and_load(self):
        save_classifier(self.clf)
        clf_loaded = load_classifier()
        X_test_transformed = preprocess_pipeline.transform(self.X_test)
        y_pred_loaded = clf_loaded.predict(X_test_transformed)
        y_pred = self.clf.predict(X_test_transformed)
        mean1 = np.mean(y_pred_loaded == self.y_test)
        mean2 = np.mean(y_pred == self.y_test)
        print("Loaded model mean:", mean1)
        print("Model mean:", mean2)
        self.assertAlmostEqual(mean1, mean2)

    # Test that saved model loads correctly and can be used for prediction
    def test_predict_probability(self):
        clf_loaded = load_classifier()
        p = round(spam_probability(clf_loaded, self.X_test[0]), 2)
        print("The predicted probability of email being spam is:", p)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)


if __name__ == '__main__':
    unittest.main()


