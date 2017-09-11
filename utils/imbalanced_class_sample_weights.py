import numpy as np


def class_weights(y_true):
    labels = np.unique(y_true).tolist()
    labels.sort()
    weights = {}
    for label in labels:
        labeled_index = y_true == label
        labeled_count = labeled_index.sum()
        weight = len(y_true) / labeled_count
        weights[label] = weight
    return weights


def class_sample_weights(y_true):
    weights = class_weights(y_true)
    result = np.zeros(y_true.shape)
    for label, weight in weights.items():
        result[y_true == label] = weight
    return result
