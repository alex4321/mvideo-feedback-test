import numpy as np
from sklearn.preprocessing import FunctionTransformer
from misc import PredictFunctionTransformer


def _conversion(X, min, max):
    X_copy = np.array(X)
    X_copy[X < min] = min
    X_copy[X > max] = max
    return X_copy


def output_range_transformation(min, max):
    return PredictFunctionTransformer(_conversion, _conversion, validate=False, kw_args={
        "min": min,
        "max": max
    })
