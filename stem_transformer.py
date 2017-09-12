from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.preprocessing import FunctionTransformer


def _stem_text(text, stemmer, tokenize):
    tokens = tokenize(text)
    return " ".join(map(stemmer.stem, tokens))


def _stem(texts, stemmer, tokenize):
    return [_stem_text(text, stemmer=stemmer, tokenize=tokenize) for text in texts]


def stem_transformer(stemmer=None, tokenize=None):
    if stemmer is None:
        stemmer = RussianStemmer()
    if tokenize is None:
        tokenize = wordpunct_tokenize
    return FunctionTransformer(_stem, validate=False, kw_args={
        "stemmer": stemmer,
        "tokenize": tokenize
    })
