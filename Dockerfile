FROM python:3

WORKDIR D:\Github\doccano_plugin

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "./retrain.py"]