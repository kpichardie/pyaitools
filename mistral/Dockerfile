FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# On définit l'entrée pour que le container se comporte comme un exécutable
ENTRYPOINT ["python", "main.py"]
