FROM python:3.8.12-buster

COPY frontend /frontend
# COPY model.joblib /model.joblib -> integrate the first base model
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn frontend.api:app --host 0.0.0.0 --port $PORT

# CMD streamlit run --server.port 8080 --server.enableCORS false app.py
