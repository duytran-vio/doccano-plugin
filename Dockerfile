FROM python:3

WORKDIR /retrain

COPY ./requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./retrain.py"]