import os

import pandas as pd

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data, NLPCleaner

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")


def test_split_data():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    assert isinstance(X, list), "X should be a list"
    assert isinstance(y, list), "y should be a list"


def test_nlp_transform():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    cleaner = NLPCleaner()

    clean_text = cleaner.transform(X)

    assert clean_text[0] == ('réserv tabl quelqu mois avanc le servic impecc pend processus nous arriv '
 'temp rapid assis un personnel accueil attent détail nous verr champagn plus')
