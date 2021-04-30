import numpy as np
import re

from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from stop_words import get_stop_words


nlp = spacy.load("fr_core_news_sm")


def split_data(data, comment_col="comment", target_col="sentiment"):
    """
    split dataset between features and label
    :param data: Pandas DataFrame
    :return: a tuple with features and label values
    """

    try:
        X = data[comment_col]

        if X.dtype != np.dtype('O'):
            raise ValueError("The comment column must have strings")

        X = list(X.values)
        y = data[target_col]

        if y.dtype not in (np.dtype("int64"), np.dtype("float64")):
            raise ValueError("The target column isn't numeric")

        y = y.values.tolist()
        return X, y
    except ValueError as e:
        raise ValueError(str(e))
    except KeyError as e:
        raise KeyError("Incorrect column names")




class NLPCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit Learn transformer class. Useful to preprocess text before applying an algorithm
    """
    def __init__(self):
        super().__init__()

    def clean_sentences(self, sentence):
        # stemming : it is done during tokenization
        stemmer = SnowballStemmer(language='french')

        def remove_stop_words(phrase):
            # tokenize
            doc = nlp(phrase)
            pattern_token = re.compile(r"(?u)\b\w\w+\b")
            tokens = [token.text for token in doc if pattern_token.match(token.text)]

            # get stop words
            french_stop_words = get_stop_words('fr')

            # remove stop_words
            clean_sentence = []
            for token in tokens:
                if token not in french_stop_words:
                    clean_sentence.append(token)
            return clean_sentence

        def stem_sentences(tokens):
            return ' '.join([stemmer.stem(token) for token in tokens])

        sent = remove_stop_words(sentence)
        return stem_sentences(sent)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = [self.clean_sentences(sent) for sent in X]
        return X_
