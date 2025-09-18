FROM runpod/pytorch:3.10-2.0.1-11.8.0

WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download Swin backbone weights
RUN mkdir -p /workspace/weights/cv
ADD https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth /workspace/weights/cv/swin_large_patch4_window12_384_22kto1k.pth

# Entrypoint for RunPod Serverless
CMD ["python3", "-u", "handler.py"]

