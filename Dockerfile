# Use an official Python runtime as a parent image
FROM python:3.12-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies first to take advantage of caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port Flask will run on
EXPOSE 5001

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production  # Set to production for better performance

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
