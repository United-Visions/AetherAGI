# perception_service/Dockerfile

# Use the official RunPod base image for PyTorch
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies required for Tesseract OCR and other libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
# First, copy the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# We need to copy the main aethermind perception/ and mind/ folders for imports
COPY . .
COPY ../perception ../perception
COPY ../mind ../mind

# Define the command to start the RunPod handler
CMD ["python", "-u", "runpod_handler.py"]
