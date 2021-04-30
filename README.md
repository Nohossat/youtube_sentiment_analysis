# Sentiment analysis API - in progress

This FastAPI application helps you infering, training and evaluating your sentiment analysis models.
You can choose the text dataset of your choice and a SVM classifier or a LGBM model to perform sentiment analysis.
The model chosen will be included in a pipeline which include a text cleaning step and a TF-IDF vectorizer.

The API collects general classification metrics if needed with the integration of Neptune.ai library.
For long-running models, you can choose to be notified when the training is over.


## How to install the API

#### Prerequisites : Environment variables

Some environment variables must be passed to make the API running.

Please replace the values in the `.env` to pass environment variables to your Docker container or your local application.

| Variable | Description |
|---------|------------|
|`LOGIN`| name used to authenticate in the API |
|`PASSWORD`| password used to authenticate in the API |
|`NEPTUNE_USER`, `NEPTUNE_PROJECT`, `NEPTUNE_API_TOKEN` | if you want to use Neptune.ai as a MLOps tool, you can provide your username, project name and API token.|
|`SENDGRID_API_KEY`, `SENDGRID_API_KEY`| if you want to enable notifications in your API, you must provide your SendGrid API KEY and the email from which the notifications will be sent from. |


### With Dockerfile

```
git clone https://github.com/Nohossat/youtube_sentiment_analysis.git
cd youtube_sentiment_analysis
docker-compose up --build
```

The application will be run on **http://0.0.0.0:5000/docs**.

### With Virtualenv

```shell
git clone https://github.com/Nohossat/youtube_sentiment_analysis.git
cd youtube_sentiment_analysis
pip install virtualenv
virtualvenv venv
source venv/bin/activate # MAC/Linux
.\venv\Scripts\activate # Windows
pip install -r requirements.txt
python -m pip install -e .
```

To get the API started :

```
cd src/nohossat_cas_pratique/
uvicorn main:app
```

The application will be run on **http://127.0.0.1:8000/docs**.

### Endpoints

|Endpoints| Description|
|---------|------------|
|`/create_user`| If you want to access the premium features below, you must create an acoount first with your username and a valid email address.|
|`/`| you can predict the polarity of several comments with a default model or a model you registered in the **models** folder|
|`/train`| you can train (with cross-validation) a SVM classifier or a LGBM model with any dataset you pass as arguments. The neptune-log set to True, sends the results to the Neptune.ai project set in your environment. You can also be notified by email when the training is done.|
|`/grid_train`| you can optimize a SVM classifier or LGBM hyperparameters. The model performance is recorded in Neptune.ai if needed. You can also be notified by email when the training is done.|
|`/models`| You can display the available models in the API. They are registered in the /models folder.|
|`/reports`| You can fetch a JSON report of the models scores recorded in Neptune.ai.|
