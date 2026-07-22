FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /alexandria

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/requirements.txt /alexandria/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt && \
    pip install --no-cache-dir qwen-tts==0.1.1

# Copy application code
COPY app/ /alexandria/app/
COPY default_prompts.txt review_prompts.txt persona_prompts.txt /alexandria/
COPY builtin_lora/ /alexandria/builtin_lora/

# Create directories for runtime data
RUN mkdir -p /alexandria/scripts \
    /alexandria/designed_voices \
    /alexandria/clone_voices \
    /alexandria/lora_models \
    /alexandria/lora_datasets \
    /alexandria/dataset_builder \
    /alexandria/app/uploads

# Bind to 0.0.0.0 inside the container
ENV ALEXANDRIA_HOST=0.0.0.0
EXPOSE 4200

CMD ["python", "app/app.py"]
