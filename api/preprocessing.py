import re

from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import spacy

nlp = spacy.load("fr_core_news_sm")

def validate_hyper(hyper_params: dict = None, default_params: list = ["C", "gamma", "kernel", "class_weight"]):
    if not isinstance(hyper_params, dict):
        raise TypeError("The hyperparameters should be in a dictionary")

    for param, values in hyper_params.items():
        if param not in default_params:
            raise ValueError("the param should be in the model params list")


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


def clean_sentences(sentence):
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
        return [stemmer.stem(token) for token in tokens]

    sent = remove_stop_words(sentence)
    return stem_sentences(sent)
