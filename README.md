# Sentiment analysis API - in progress

This FastAPI application helps you infering, training and evaluating your sentiment analysis models.
You can choose the text dataset of your choice and a valid Scikit-Learn Classifier to perform sentiment analysis.
You can also collect the classification metrics in any Neptune.ai project.


## How to install the API

### With DockerHub - NOT WORKING YET

```shell
docker pull nohossat1/sentiment-analysis-api
docker run -it -p 5000:5000 --name sentiment_analysis_api nohossat1/sentiment-analysis-api:latest
```

### With Dockerfile

```
git clone https://github.com/Nohossat/youtube_sentiment_analysis.git
cd youtube_sentiment_analysis
docker build -t sentiment-analysis .
docker run -p 5000:8000 --name sentiment-analysis-api sentiment-analysis:latest
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

## How to link the API to your Neptune.ai account

If you install this project with the Dockerfile, you can set the NEPTUNE_USER, NEPTUNE_PROJECT and NEPTUNE_TOKEN to your own.
If they are correct, when running an training job, the data will be recorded in your project.


## Usage

To get the API started :

```
cd src/nohossat_cas_pratique/
uvicorn main:app
```

### Endpoints

|Endpoints| Description|
|---------|------------|
|/| you can predict the polarity of a comment with a default model or a model you registered in the **models** folder|
|/train| you can train (with cross-validation) the Scikit-Learn Classifier of your choice and any dataset you pass as arguments. The neptune-log set to True, sends the results to the Neptune.ai project set in your environment|
|/grid_train||
|/models||
|/reports||
