from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge


class TransformPredictorMixin:
    def transform(self, X):
        return self.predict(X)


class PredictTransformerMixin:
    def predict(self, X):
        return self.transform(X)


class TransformRidge(Ridge, TransformPredictorMixin):
    pass


class PredictFunctionTransformer(FunctionTransformer, PredictTransformerMixin):
    pass