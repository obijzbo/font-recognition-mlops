FROM tensorflow/tensorflow:2.18.0

ENV APP_PORT=${APP_PORT}

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app-req.txt .
COPY lenet5_model.keras .
COPY class_labels.json .
COPY main.py .
COPY tasks.py .
COPY celery_worker.py .

RUN pip install --no-cache-dir -r app-req.txt

EXPOSE ${APP_PORT}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
