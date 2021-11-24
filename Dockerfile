FROM python:3.8.12-buster

COPY api /api
# COPY model.joblib /model.joblib -> integrate the first base model
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
