# Use the specified PyTorch base image with CUDA support
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime AS xmem2-base-inference

# Set the working directory in the container
WORKDIR /app

RUN python -m pip install --no-cache-dir opencv-python-headless Pillow==9.2.0

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . /app

# FOR GUI - only a few extra dependencies
FROM xmem2-base-inference AS xmem2-gui

# Qt dependencies
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 libxkbcommon-x11-0 '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libfontconfig libdbus-1-3 mesa-utils libgl1-mesa-glx
RUN /bin/bash -c 'gcc --version'

RUN python -m pip install --no-cache-dir -r requirements_demo.txt
# To avoid error messages when launching PyQT
ENV LIBGL_ALWAYS_INDIRECT=1