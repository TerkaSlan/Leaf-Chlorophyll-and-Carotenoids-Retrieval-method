# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y tmux zip wget && \
    apt-get clean

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run when the container starts
CMD ["tmux"]