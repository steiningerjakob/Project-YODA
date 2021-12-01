FROM python:3.8.12-buster

COPY api /api
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY projectYoda /projectYoda
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
