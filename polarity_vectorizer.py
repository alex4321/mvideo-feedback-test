import numpy as np
from sklearn.preprocessing import FunctionTransformer
from polyglot.text import Text


INVERSE_TOKEN = "не"


def _polarity_vector(polarities):
    positive_count = (polarities > 0).sum()
    negative_count = (polarities < 0).sum()
    vector_non_norm_2d = np.array([positive_count, negative_count])
    length = np.linalg.norm(vector_non_norm_2d)
    if length == 0:
        return np.array([0.0, 0.0, 1.0])
    else:
        vector_norm_2d = (vector_non_norm_2d / length) ** 2
        neutral = 1.0 - vector_norm_2d.sum()
        return np.array([vector_norm_2d[0], vector_norm_2d[1], neutral])


def _text_polarity(text, language):
    tokens = list(Text(text.lower(), hint_language_code=language).words)
    tokens_shifted = tokens[1:] + [""]
    polarities = []
    skip = False
    for token, next_token in zip(tokens, tokens_shifted):
        if skip:
            skip = False
            continue
        polarity = token.polarity
        if str(token) == INVERSE_TOKEN:
            polarity = -next_token.polarity
            skip = True
        polarities.append(polarity)
    return _polarity_vector(np.array(polarities))


def _texts_polarities(texts, language):
    return [_text_polarity(text, language) for text in texts]


def polarity_vectorizer(language):
    return FunctionTransformer(_texts_polarities, validate=False, kw_args={
        "language": language
    })