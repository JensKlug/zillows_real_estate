# Use official lightweight Python image
FROM python:3.10.6-buster

# Set working directory in container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of the app source code
COPY . .
COPY raw_data raw_data

# Expose port (default 8000)

# Start FastAPI app with uvicorn
# CMD ["uvicorn", "zillow.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn zillow.api.fast:app --host 0.0.0.0 --port $PORT
