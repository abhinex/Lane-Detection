# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install any dependencies specified directly
RUN pip install --no-cache-dir Flask opencv-python-headless numpy

# Copy the rest of the application code into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
