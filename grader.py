import unittest
import numpy as np
import time
# import timeout_decorator
from scipy.sparse import csc_matrix
from decision_tree import *
from naive_bayes import *
from crossval import *
from load_all_data import load_all_data


class GraderTestCase(unittest.TestCase):
	@staticmethod
	def unique_data():
		"""Set up uniquely identifiable examples"""
		train_data = np.zeros((4, 10))

		# label = 0
		train_data[:, 0] = [0, 0, 0, 0]
		train_data[:, 1] = [0, 0, 0, 1]
		train_data[:, 2] = [0, 0, 1, 0]
		train_data[:, 3] = [0, 0, 1, 1]
		train_data[:, 4] = [0, 1, 0, 0]

		# label = 1
		train_data[:, 5] = [0, 1, 0, 1]
		train_data[:, 6] = [0, 1, 1, 0]
		train_data[:, 7] = [0, 1, 1, 1]
		train_data[:, 8] = [1, 0, 0, 0]
		train_data[:, 9] = [1, 0, 0, 1]

		train_data = csc_matrix(train_data)

		train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

		return train_data, train_labels

	@staticmethod
	def real_data():
		"""Load 20 newsgroups data"""
		num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

		return train_data, train_labels

	# @timeout_decorator.timeout(1)
	def test_decision_tree_memorization(self):
		"""Test whether a decision tree can memorize unique data"""
		train_data, train_labels = self.unique_data()
		# train_data, train_labels = self.real_data()

		# set tree depth to unlimited
		params = {"max_depth": np.inf}

		model = decision_tree_train(train_data, train_labels, params)

		predictions = decision_tree_predict(train_data, model)

		assert np.all(predictions == train_labels), "Decision tree was unable to memorize 10 unique examples"

	# @timeout_decorator.timeout(10, exception_message="Training too slow.")
	def test_decision_tree_speed(self):
		"""Check that the decision tree implementation is fast."""
		n = 50000
		d = 100
		train_data = csc_matrix(np.random.randint(0, 1, size=(d, n)))
		train_labels = np.random.randint(0, 5, size=n)

		start_time = time.time()
		params = {"max_depth": 10}
		model = decision_tree_train(train_data, train_labels, params)
		predictions = decision_tree_predict(train_data, model)
		end_time = time.time()

		print("Decision tree training and prediction took %f seconds." % (end_time - start_time))

		assert (end_time - start_time) < 10.0, "Training on a (somewhat) large dataset took longer than 10 seconds."

	# @timeout_decorator.timeout(10)
	def test_decision_tree_depth(self):
		"""Test that max_depth is implemented correctly."""
		train_data, train_labels = self.unique_data()

		expected_correct = [5, 7, 9, 9, 10]

		for depth in range(4):
			params = {"max_depth": depth}
			model = decision_tree_train(train_data, train_labels, params)
			predictions = decision_tree_predict(train_data, model)

			# Accuracy should be 0.5 at depth 0.
			num_correct = np.sum(predictions == train_labels)
			self.assertEqual(num_correct, expected_correct[depth], ("Incorrect accuracy for depth %d." % depth))

	# @timeout_decorator.timeout(10)
	def test_decision_tree_improvement(self):
		"""Test that deeper decision trees have better training performance on real data"""
		train_data, train_labels = self.real_data()

		params = {"max_depth": 0}
		shallow_tree = decision_tree_train(train_data, train_labels, params)
		shallow_predictions = decision_tree_predict(train_data, shallow_tree)

		for depth in range(1, 5):
			params = {"max_depth": depth}

			deep_tree = decision_tree_train(train_data, train_labels, params)
			deep_predictions = decision_tree_predict(train_data, deep_tree)

			num_correct_shallow = np.sum(shallow_predictions == train_labels)
			num_correct_deep = np.sum(deep_predictions == train_labels)

			print("Shallow tree correctly labels %d training examples" % num_correct_shallow)
			print("Deep tree correctly labels %d training examples" % num_correct_deep)

			assert num_correct_deep > num_correct_shallow, "Deep tree didn't improve training accuracy over shallow tree."

			shallow_tree = deep_tree
			shallow_predictions = deep_predictions

	# @timeout_decorator.timeout(60, exception_message="Decision tree took too long on real data.")
	def test_decision_tree_test_accuracy(self):
		"""Test that decision tree accuracy is sufficiently high."""
		num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

		params = {"max_depth": 16}

		model = decision_tree_train(train_data, train_labels, params)
		predictions = decision_tree_predict(test_data, model)

		accuracy = np.sum(predictions == test_labels) / num_testing

		print("Decision tree accuracy was %f" % accuracy)

		assert accuracy > 0.3, "Decision tree accuracy was lower than expected."

	# @timeout_decorator.timeout(10)
	def test_naive_bayes_simple(self):
		"""Test naive Bayes on a tiny toy problem."""
		train_data = np.zeros((4, 4))

		# negative examples
		train_data[:, 0] = [1, 0, 0, 0]
		train_data[:, 1] = [0, 1, 0, 0]

		# positive examples
		train_data[:, 2] = [0, 0, 1, 0]
		train_data[:, 3] = [0, 0, 0, 1]

		train_data = csc_matrix(train_data)

		train_labels = np.array([0, 0, 1, 1])

		model = naive_bayes_train(train_data, train_labels, {"alpha": 0})

		predictions = naive_bayes_predict(train_data, model)

		accuracy = np.mean(predictions == train_labels)

		print("Accuracy on toy data was %f" % accuracy)

		self.assertEqual(accuracy, 1.0)

	# @timeout_decorator.timeout(30, exception_message="Naive Bayes took too long on real data.")
	def test_naive_bayes_accuracy(self):
		"""Test that decision tree accuracy is sufficiently high."""
		num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

		params = {"alpha": 1e-4}

		model = naive_bayes_train(train_data, train_labels, params)
		predictions = naive_bayes_predict(test_data, model)

		accuracy = np.mean(predictions == test_labels)

		print("Naive Bayes accuracy was %f" % accuracy)

		assert accuracy > 0.7, "Naive Bayes accuracy was lower than expected."

	# @timeout_decorator.timeout(1, exception_message="Training too slow.")
	def test_naive_bayes_speed(self):
		"""Check that the decision tree implementation is fast."""
		n = 50000
		d = 100
		train_data = csc_matrix(np.random.randint(0, 1, size=(d, n)))
		train_labels = np.random.randint(0, 5, size=n)

		start_time = time.time()
		params = {"alpha": 0.1}
		model = naive_bayes_train(train_data, train_labels, params)
		predictions = naive_bayes_predict(train_data, model)
		end_time = time.time()

		print("Naive Bayes training and prediction took %f seconds." % (end_time - start_time))

		assert (end_time - start_time) < 1.0, "Training on a (somewhat) large dataset took longer than a second."

	@staticmethod
	def dummy_learner(data, labels, params):
		"""Empty learner for testing cross-validation."""
		return None

	@staticmethod
	def dummy_predictor(data, model):
		"""Empty predictor for testing cross-validation."""
		return np.zeros(data.shape[1])

	# @timeout_decorator.timeout(1, exception_message="Cross-validation too slow.")
	def test_crossval_speed(self):
		"""Test speed of data splitting for cross validation."""
		n = 50000
		d = 100
		train_data = csc_matrix(np.random.randint(0, 1, size=(d, n)))
		train_labels = np.random.randint(0, 5, size=n)

		start_time = time.time()
		score, models = cross_validate(self.dummy_learner, self.dummy_predictor, train_data, train_labels, 10, {})
		print("Finished doing data splitting for cross validation in %f seconds." % (time.time() - start_time))

		assert (time.time() - start_time) < 1.0, "Cross-validation too slow."

	# @timeout_decorator.timeout(60)
	def test_crossval_correctness(self):
		"""Test that cross validation returns the correct number of scores."""
		train_data, train_labels = self.real_data()

		for folds in [5, 10, 20]:
			score, models = cross_validate(self.dummy_learner, self.dummy_predictor, train_data, train_labels, folds, {})
			self.assertEqual(len(models), folds)

	@staticmethod
	# @timeout_decorator.timeout(60)
	def test_crossval_naive_bayes():
		"""Test that cross-validation on naive Bayes gives scores close to held-out test accuracy"""
		num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

		params = {"alpha": 1e-4}

		score, models = cross_validate(naive_bayes_train, naive_bayes_predict, train_data, train_labels, 3, params)

		print("Cross-validation score was %f." % score)

		model = naive_bayes_train(train_data, train_labels, params)
		predictions = naive_bayes_predict(test_data, model)

		accuracy = np.mean(predictions == test_labels)

		print("Naive Bayes accuracy was %f" % accuracy)

		assert np.abs(accuracy - score) < 0.2, "Accuracy and cross-validation score were not close."


if __name__ == '__main__':
	unittest.main()
