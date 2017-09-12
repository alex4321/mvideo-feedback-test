from nltk.tokenize import wordpunct_tokenize
from sklearn.preprocessing import FunctionTransformer


INVERSE_TOKEN = "не"


def _inverse_token_concatenate_text(text, tokenize):
    tokens = tokenize(text.lower())
    tokens_shifted = tokens[1:] + [""]
    result = []
    skip = False
    for token, next_token in zip(tokens, tokens_shifted):
        if skip:
            skip = False
            continue
        token_result = token
        if token == "не" and next_token != "":
            token_result = token + "_" + next_token
            skip = True
        result.append(token_result)
    return " ".join(result)


def _inverse_token_concatenate_texts(texts, tokenize):
    return [_inverse_token_concatenate_text(text, tokenize) for text in texts]


def inverse_token_concatenation(tokenize=None):
    if tokenize is None:
        tokenize = wordpunct_tokenize
    return FunctionTransformer(_inverse_token_concatenate_texts, validate=False, kw_args={
        "tokenize": tokenize
    })