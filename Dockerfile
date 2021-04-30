FROM python:3.8

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN mkdir /api
RUN mkdir /api/data
RUN mkdir /api/src
RUN mkdir /api/models

WORKDIR /api

COPY setup.cfg .
COPY setup.py .
COPY data/comments.csv ./data
COPY data/users.db ./data
COPY models/sentiment_pipe.joblib ./models
COPY models/grid_search_SVC.joblib ./models
COPY models/SVC_solo.joblib ./models
COPY src ./src
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN python -m pip install -e .

EXPOSE 8000

WORKDIR /api/src/nohossat_cas_pratique

CMD "uvicorn" "main:app" "--host" "0.0.0.0" "--port" $PORT
