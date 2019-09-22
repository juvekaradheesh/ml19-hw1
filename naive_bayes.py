"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
from load_all_data import load_all_data
import math

def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    model={}
        
    train_labels = [int(x) for x in train_labels]

    val={}
    ct=np.zeros(num_classes)
    train_data = np.transpose(train_data)
    for i in range(len(train_labels)):
        if train_labels[i] in val:
            old_val = val.pop(train_labels[i])
            val[train_labels[i]]=[x + y for x, y in zip(old_val,train_data[i])]
            ct[train_labels[i]]+=1
        else:
            val[train_labels[i]]=train_data[i]+alpha
            ct[train_labels[i]]=1

    for k,v in val.items():
        model[k]=np.true_divide(v,(ct[k]+2*alpha))
    
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    
    d, n = data.shape
    nclass = (len(model))
    predicted_class = np.zeros(n);
    for i in range(n):
        
        test_data = data[:,i]
        prob = np.zeros(nclass)
        
        for k in range(len(model)) : 
            index1 = np.where(test_data != 0)
            index0 = np.where(test_data == 0)
            feature_prob = model[k]
            total_prob1 = np.sum(np.log(feature_prob[index1]))
            prob0 = np.ones(len(index1)) - feature_prob[index0]
            total_prob0 = np.sum(np.log(prob0))
            prob[k] = total_prob1 + total_prob0
        
        predicted_class[i] = np.argmax(prob)

    return predicted_class

