# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /MedStrips

RUN apt-get update && apt-get install -y \
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
EXPOSE 5000

# Define the command to run your application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=5000", "--server.address=127.0.0.1"]