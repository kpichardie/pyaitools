FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY gemini_tool.py .

# Set the entrypoint so you can just pass arguments directly
ENTRYPOINT ["python", "gemini_tool.py"]
