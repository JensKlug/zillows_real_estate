# Use a lightweight Python image
FROM python:3.10.6-buster

# Set the working directory in the container
WORKDIR /app

# Copy everything except what's in .dockerignore
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the port
ENV PORT=8000
EXPOSE $PORT

# Start the FastAPI app (adjust path if needed)
CMD ["sh", "-c", "uvicorn zillow.api.fast:app --host 0.0.0.0 --port ${PORT:-8000}"]
