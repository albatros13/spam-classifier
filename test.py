import unittest
import numpy as np
from main import load_data, train_classifier, spam_probability, save_classifier, load_classifier
from transformer import preprocess_pipeline
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

        self.clf = train_classifier(self.X_train, self.y_train)
        file_name = "saved_model.pk1"
        save_classifier(self.clf, file_name)
        print('The classifier has been trained and saved in a file: {}'.format(file_name))

    # Test classifier accuracy
    # @unittest.skip("Skipping performance test")
    def test_classifier(self):
        X_test_transformed = preprocess_pipeline.transform(self.X_test)
        y_pred = self.clf.predict(X_test_transformed)
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
        X_test_transformed = preprocess_pipeline.transform(self.X_test)
        y_pred_loaded = clf_loaded.predict(X_test_transformed)
        y_pred = self.clf.predict(X_test_transformed)
        mean1 = np.mean(y_pred_loaded == self.y_test)
        mean2 = np.mean(y_pred == self.y_test)
        self.assertAlmostEqual(mean1, mean2)

    # Test that saved model can be used for prediction
    # @unittest.skip("Skipping prediction test")
    def test_predict_probability(self):
        clf_loaded = load_classifier()
        p = round(spam_probability(clf_loaded, self.X_test[0]), 2)
        # print("The predicted probability of the email being spam is:", p)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)


if __name__ == '__main__':
    unittest.main()


