# Sentiment analysis API

This FastAPI application helps you infering, training and evaluating your sentiment analysis models.

## How to install

### With Docker

```shell
docker pull nohossat1/sentiment-analysis-api
docker run -it -p 5000:5000 --name sentiment_analysis_api nohossat1/sentiment-analysis-api:latest
```

### With Virtualenv

```shell
pip install virtualenv
git clone URL
cd DOSSIER
virtualvenv
source venv/bin/activate
pip install -r requirements.txt
```

To get the API started :

```
uvicorn main:app
```

{'clf__C': [0.1, 5, 10, 100, 500, 1000],
                                'clf__class_weight': ['balanced', None, {0: 0.37, 1: 0.63}],
                                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                'clf__gamma': [0.001, 0.01, 0.1, "scale", "auto"]}
