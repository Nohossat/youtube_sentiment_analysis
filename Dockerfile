FROM python:3.9

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN mkdir /api
RUN mkdir /api/data
RUN mkdir /api/nohossat_cas_pratique

WORKDIR /api

COPY setup.cfg .
COPY setup.py .
COPY data/comments.csv ./data
COPY src/nohossat_cas_pratique ./nohossat_cas_pratique
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python -m pip install -e .

EXPOSE 8000

CMD "uvicorn" "main:app" "--reload" "--host" "0.0.0.0" "--port" "8000"
