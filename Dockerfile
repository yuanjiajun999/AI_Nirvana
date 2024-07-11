FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install transformers scikit-learn

COPY ./src /app/src
RUN ls -la /app/src