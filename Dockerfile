FROM python:3.8

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN mkdir /api

WORKDIR /api

COPY main.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD "uvicorn" "main:app" "--reload" "--host" "0.0.0.0" "--port" "8000"
