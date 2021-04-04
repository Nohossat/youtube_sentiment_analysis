# Sentiment analysis API - in progress

This FastAPI application helps you infering, training and evaluating your sentiment analysis models.
You can choose the text dataset of your choice and a SVM classifier or a LGBM model to perform sentiment analysis.
The model chosen will be included in a pipeline which include a text cleaning step and a TF-IDF vectorizer.

The API collect general classification metrics if needed with the integration of Neptune.ai library.
For long-running models, you can choose to be notified when the training is over.


## How to install the API

#### Environment variables

Some environment variables must be passed to make the API running.

You can set them in the Dockerfile, if you want to run the API inside a container.
Or set them in your OS.

**LOGIN** : name used to authenticate in the API
**PASSWORD** : password used to authenticate in the API
**NEPTUNE_USER**, **NEPTUNE_PROJECT**, **NEPTUNE_API_TOKEN** : if you want to use Neptune.ai as a MLOps tool, you can provide your username, project name and API token.
**SENDGRID_API_KEY**, **SENDGRID_API_KEY**: if you want to enable notifications in your API, you must provide your SendGrid API KEY and the email from which the notifications will be sent from.

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


## Usage

To get the API started :

```
cd src/nohossat_cas_pratique/
uvicorn main:app
```

### Endpoints

|Endpoints| Description|
|---------|------------|
|`/`| you can predict the polarity of several comments with a default model or a model you registered in the **models** folder|
|`/train`| you can train (with cross-validation) a SVM classifier or a LGBM model with any dataset you pass as arguments. The neptune-log set to True, sends the results to the Neptune.ai project set in your environment. You can also be notified by email when the training is done.|
|`/grid_train`| you can optimize a SVM classifier or LGBM hyperparameters. The model performance is recorded in Neptune.ai if needed. You can also be notified by email when the training is done.|
|`/models`| You can display the available models in the API. They are registered in the /models folder.|
|`/reports`| You can fetch a JSON report of the models scores recorded in Neptune.ai.|
