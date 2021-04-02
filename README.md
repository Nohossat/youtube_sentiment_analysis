# Sentiment analysis API - in progress

This FastAPI application helps you infering, training and evaluating your sentiment analysis models.

## How to install

### With Docker - NOT WORKING YET

```shell
docker pull nohossat1/sentiment-analysis-api
docker run -it -p 5000:5000 --name sentiment_analysis_api nohossat1/sentiment-analysis-api:latest
```

### With Virtualenv

```shell
pip install virtualenv
git clone https://github.com/Nohossat/youtube_sentiment_analysis.git
cd youtube_sentiment_analysis
virtualvenv
source venv/bin/activate
pip install -r requirements.txt
python -m pip install -e .
```

To get the API started :

```
cd src/nohossat_cas_pratique/
uvicorn main:app
```
