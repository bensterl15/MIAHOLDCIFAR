# Step 1: Choose a base image with CUDA 11.1 and cuDNN 8.
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Step 2: Set environment variables for the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENV WANDB_API_KEY=2862a5eb3fff995e267ad1ef705e0a11307595f5

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Install pip and any other system-level dependencies needed for Python packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Step 5: Copy your requirements.txt and install Python dependencies (excluding JAX for now)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu111

# --- NEW STEP 6: Install jax and jaxlib explicitly ---
# This ensures that jaxlib is installed with the correct CUDA/cuDNN backend.
RUN pip install --no-cache-dir \
    jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Step 7: Copy the rest of your project code into the container (now was Step 6)
COPY . .


# Step 8: Run training directly (no start_with_tb.sh)
CMD ["python3", "main.py", \
     "-cc", "configs/default_cifar10.txt", \
     "-sc", "configs/specific_cifar10.txt", \
     "--root", ".", \
     "--mode", "train", \
     "--workdir", "work_dir/cifar", \
     "--n_gpus_per_node", "1", \
     "--training_batch_size", "8", \
     "--testing_batch_size", "8", \
     "--sampling_batch_size", "32", \
     "--eval_fid_samples", "128", \
     "--master_port", "6026"]
