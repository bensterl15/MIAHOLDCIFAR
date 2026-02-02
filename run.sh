docker run -d \
           --gpus "device=0" \
           --name nold \
           -p 6006:6006 \
           -v $(pwd):/app \
           -v $(pwd)/output_data:/app/work_dir/cifar \
           nold:latest
