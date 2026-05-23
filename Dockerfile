# Use an official lightweight Python image
FROM python:3.10-slim

# Install system-level C++ dependencies for OpenCV, Audio, and TF Flex Delegate
# FIX: Updated obsolete 'libgl1-mesa-glx' to 'libgl1' for the latest Debian slim
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (Required by Hugging Face Docker Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy your requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY --chown=user . .

# Expose the Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]