FROM python:3.9

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN mkdir /api
RUN mkdir /api/data
RUN mkdir /api/src

WORKDIR /api

COPY setup.cfg .
COPY setup.py .
COPY data/comments.csv ./data
COPY src ./src
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python -m pip install -e .

EXPOSE 8000

WORKDIR /api/src/nohossat_cas_pratique

CMD "uvicorn" "main:app" "--host" "0.0.0.0" "--port" "8000"
