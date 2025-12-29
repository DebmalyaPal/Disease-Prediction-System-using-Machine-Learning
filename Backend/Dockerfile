# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better for Docker build caching)
COPY requirements.txt .

# Install system dependencies if needed (e.g., for pandas, flask)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .
COPY models ./models
COPY Data ./Data

# set environment variables
ENV FLASK_APP=app.py
# ENV FLASK_ENV=production

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
