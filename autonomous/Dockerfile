FROM python:3.11-slim

# Install system dependencies including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --progress-bar off \
    google-genai \
    python-telegram-bot \
    python-telegram-bot[job-queue] \
    httpx

# Create necessary directories
RUN mkdir -p /app/backups

# Copy application code
COPY autonomous_orchestrator.py .
COPY project-context.md .

# Initialize git repository for self-updates
RUN git init && \
    git config user.email "agent@autonomous.local" && \
    git config user.name "Autonomous Agent" && \
    git add . && \
    git commit -m "Initial commit - v2.0.0"

# Set environment defaults
ENV DB_FILE=/app/tasks.db
ENV LOG_FILE=/app/autonomous.log
ENV MODEL_ID=gemini-2.0-flash-exp
ENV MAX_WORKERS=3
ENV RETRY_DELAY_MINUTES=5
ENV MAX_RETRIES=3
ENV OLLAMA_HOST=http://localhost:11434
ENV OLLAMA_MODEL=llama3
ENV FORCED_AI_REPORT=false

# Volume for persistent data
VOLUME ["/app/backups"]

# Run the agent
CMD ["python", "autonomous_orchestrator.py"]
