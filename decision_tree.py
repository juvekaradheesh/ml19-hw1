"""This module includes methods for training and predicting using decision trees."""
import numpy as np
from tree import Node


def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)

    # print("Full entropy is %d\n" % full_entropy)

    gain = full_entropy * np.ones(d)

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n))
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    
    # Create object of Node class from tree.py
    node = Node()
    
    # If labels have only one class, make that node as leaf node
    if len(np.unique(labels)) == 1:
        
        node.prediction = labels[0]
        return node

    # If max depth limit reached, make that node as leaf node and 
    # predict the class with highest probability(count)
    if depth >= max_depth:

        # Count occurence of each class in current subtree data.
        class_count = np.zeros(num_classes)

        for c in range(num_classes):
            class_count[c] = np.sum(labels == c)

        # Make prediction of classes with maximum probability(count)
        node.prediction = np.argmax(class_count)
        
        return node
    
    # If labels are more that 2 and depth is less than maximum depth limit 
    # then proceed

    # calculate information gain for all attributes
    gain = calculate_information_gain(data,labels)
    
    # Set test as the attribute with highest Information Gain
    node.test = np.argmax(gain)
    
    # The following code splits the data and labels for left and right subtree
    split_indices = np.where(data[node.test])
    temp_data = data.transpose()
    mask = np.ones(len(temp_data), dtype=bool)
    mask_labels = np.ones(len(labels), dtype=bool)
    mask[split_indices,] = False
    left_data, right_data = temp_data[split_indices].transpose(), temp_data[mask].transpose()
    left_labels, right_labels = labels[split_indices], labels[mask]
    
    
    # Recursively call left subtree and right subtree
    node.left_child = recursive_tree_train(left_data, left_labels, depth+1, max_depth, num_classes) 
    node.right_child = recursive_tree_train(right_data, right_labels, depth+1, max_depth, num_classes)

    return node


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """

    predict_labels = np.zeros(len(data.T))
    curr_node = model

    # The for loop iterates through the data, per data row.
    for i, curr_data in enumerate(data.T):
        
        # Add prediction value of current leaf node to the prediction list
        predict_labels[i] = get_prediction(curr_data, model)
        curr_node.prediction

    return predict_labels

def get_prediction(data, model):
    
    # If a leaf node is found, predict
    if model.prediction != None:
        return model.prediction
    
    # If leaf node not found -> recursive call to left or right subtree
    if data[model.test]:
        return get_prediction(data, model.left_child)
    else:
        return get_prediction(data,model.right_child)
