# perception_service/Dockerfile

# Use the official RunPod base image for PyTorch
FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

# Set the working directory
WORKDIR /app

# Install system dependencies required for Tesseract OCR and other libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
# First, copy the requirements file to leverage Docker layer caching
COPY perception_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# We need to copy the main aethermind perception/ and mind/ folders for imports
COPY perception_service/ ./
COPY perception ./perception
COPY mind ./mind

# Define the command to start the RunPod handler
CMD ["python", "-u", "runpod_handler.py"]
