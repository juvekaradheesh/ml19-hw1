"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))

    models = []
    
    # print("Total folds: ", folds)
    # print("-"*30)
    for cover_fold in range(folds):
        train_indices = np.delete(indices,cover_fold,1).flatten()
        validation_indices = indices[:,cover_fold]
        models.append(trainer(all_data[:,train_indices], all_labels[train_indices], params))
        val_predictions = predictor(all_data[:,validation_indices], models[cover_fold])
        scores[cover_fold] = np.mean(val_predictions == all_labels[validation_indices])
        # print("Training and prediction complete" )
        # print("Fold: ", cover_fold + 1, "Score: ",scores[cover_fold])
        # print("-"*30)
        
    score = np.mean(scores)

    return score, models
