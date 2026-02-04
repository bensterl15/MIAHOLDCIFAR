docker run -d \
           --gpus "device=0" \
           --name noldfid \
           -p 6007:6007 \
           -v $(pwd):/app \
           -v $(pwd)/output_data:/app/work_dir/celeba64 \
           noldfid:latest
