import unittest
import numpy as np
from classifier import load_data, save_classifier, load_classifier, pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class TestClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestClassifier, self).setUpClass()

        X, y = load_data()
        print('Loaded {} emails and {} labels'.format(len(X), len(y)))

        # train_test_split may be slow, to avoid calling it for each test we store data in class variables
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('Split data into {} training samples and {} testing samples'.format(len(self.X_train), len(self.X_test)))

        self.clf = pipeline.fit(self.X_train, self.y_train)
        save_classifier(self.clf)

    # Test classifier performance
    # @unittest.skip("Skipping performance test")
    def test_classifier(self):
        y_pred = self.clf.predict(self.X_test)
        precision = 100 * precision_score(self.y_test, y_pred)
        recall = 100 * recall_score(self.y_test, y_pred)
        f1 = 100 * f1_score(self.y_test, y_pred)

        print("Performance metrics:")
        print("Precision: {:.2f}%".format(precision))
        print("Recall: {:.2f}%".format(recall))
        print("F1: {:.2f}%".format(f1))

        self.assertGreaterEqual(precision, 80)
        self.assertGreaterEqual(recall, 80)

    # Test that the saved model is loaded correctly
    # @unittest.skip("Skipping load test")
    def test_load(self):
        clf_loaded = load_classifier()
        y_pred_loaded = clf_loaded.predict(self.X_test)
        y_pred = self.clf.predict(self.X_test)
        mean1 = np.mean(y_pred_loaded == self.y_test)
        mean2 = np.mean(y_pred == self.y_test)
        self.assertEqual(mean1, mean2)

    # Test that saved model can be used for prediction
    # @unittest.skip("Skipping prediction test")
    def test_predict_probability(self):
        clf_loaded = load_classifier()
        email = "Dear Winner, we wish to congratulate and inform you that your email address has won ($2,653,000 two million six hundred and fifty three thousand US Dollars)"
        p = clf_loaded.predict_proba([email])
        p = round(p[0][1],2)
        print("The predicted probability of the email being spam is:", p)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)


if __name__ == '__main__':
    unittest.main()


