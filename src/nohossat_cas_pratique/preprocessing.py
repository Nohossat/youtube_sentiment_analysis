import re

from sklearn.base import BaseEstimator, TransformerMixin
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import spacy

nlp = spacy.load("fr_core_news_sm")


def split_data(data):
    """
    split dataset between features and label
    :param data: Pandas DataFrame
    :return:
    """
    target = "sentiment"
    X = data.drop(target, axis = 1)
    X = [comment[0] for comment in X.values]
    y = data[target]
    y = y.values.tolist()
    return X, y


class NLPCleaner(BaseEstimator, TransformerMixin):
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
