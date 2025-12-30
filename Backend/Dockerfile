###############################
# Stage 1 — Build / Training
###############################
FROM python:3.9-slim AS builder

WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install dependencies including Jupyter (needed only for build stage)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyter

# Copy the relevant part of the project
COPY Data ./Data
COPY MLcode_Notebook.ipynb .
COPY disease_ensemble.py .

# Create model directory
RUN mkdir -p ./Model

# Convert notebook → script
RUN jupyter nbconvert --to script MLcode_Notebook.ipynb --output notebook

# Run the script to generate model + json data files
RUN python notebook.py

# At this point, Model/ contains trained model artifacts and Data/ contains Disease & Symptoms Info JSON files.


###############################
# Stage 2 — Flask App Image
###############################
FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY custom_exceptions.py .
COPY disease_ensemble.py .
COPY logging_config.py .

RUN mkdir -p ./Data

# Copy model and data artifacts from builder stage
COPY --from=builder /app/Model ./Model
COPY --from=builder /app/Data/Disease_Info.json ./Data/Disease_Info.json
COPY --from=builder /app/Data/Symptoms_Info.json ./Data/Symptoms_Info.json

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose Flask port
EXPOSE 5000

# Run the Flask app
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
CMD ["python", "app.py"]
