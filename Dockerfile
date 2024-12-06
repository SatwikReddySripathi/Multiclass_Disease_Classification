FROM apache/airflow:2.5.1-python3.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install pyopenssl==23.2.0 cryptography==39.0.0