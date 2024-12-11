# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /MedStrips

ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY="*"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    software-properties-common

# Copy the requirements file into the container
COPY requirements-dev.txt .

RUN python -m pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements-dev.txt

# Copy all project files into the container
COPY . .

# Expose the port your application will run on
EXPOSE 8000

# Define the command to run your application
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:8000", "app:app"]