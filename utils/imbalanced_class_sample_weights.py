import numpy as np


def imbalanced_class_sample_weights(y_true):
    labels = np.unique(y_true).tolist()
    labels.sort()
    weights = np.zeros(y_true.shape)
    for label in labels:
        labeled_index = y_true == label
        labeled_count = labeled_index.sum()
        label_weight = len(y_true) / labeled_count
        weights[labeled_index] = label_weight
    return weights
