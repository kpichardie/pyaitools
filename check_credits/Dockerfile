FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir rich requests

# Run as non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

COPY check_credits.py .

CMD ["python", "check_credits.py"]
