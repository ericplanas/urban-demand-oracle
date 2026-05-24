FROM python:3.11-slim

WORKDIR /app

COPY services/etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/etl/ ./services/etl/
RUN touch /app/services/__init__.py

ENV PYTHONPATH=/app:/app/services/etl

CMD ["python", "-m", "services.etl.main"]
