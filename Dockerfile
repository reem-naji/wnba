FROM python:3.12-slim

WORKDIR /mvp-predictor

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r ./requirements.txt

COPY ml/models/ ./ml/models

COPY dashboard/ ./dashboard

COPY ml/data/ ./ml/data

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]