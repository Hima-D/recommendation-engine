# Use official slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Upgrade pip and install system dependencies
# libgcc and others are often needed for scientific packages like numpy/scipy/torch
RUN apt-get update && \
    apt-get install -y build-essential gcc libffi-dev curl && \
    pip install --upgrade pip

# Copy only requirements.txt to cache Docker layers
COPY requirements.txt .

# Install dependencies with retries and extended timeout
RUN pip install --default-timeout=100 --retries=10 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI app will run on
EXPOSE 8000

# Set the default command to run your API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
