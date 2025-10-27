# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by opencv-python-headless
# libgl1 is often needed for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Upgrade pip first, pin numpy, install the rest
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 10000 available to the world outside this container
# Render's default port, matches Gunicorn default if not specified
EXPOSE 10000

# Define environment variable if needed (though Gunicorn uses PORT env var)
# ENV PORT=10000

# Run app.py when the container launches using Gunicorn
# Use 0.0.0.0 to bind to all interfaces
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]