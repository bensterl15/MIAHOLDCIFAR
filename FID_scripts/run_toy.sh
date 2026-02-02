export CUDA_VISIBLE_DEVICES=1

dir="toy_work_dir/toy_seed_0/"

# Check if directory exists:
if [ -d "$dir" ]; then
	# If directory exists, remove it:
	rm -r "$dir"
	echo "Directory $dir removed."
else
	# Directory does not exist
	echo "Directory $dir does not exist."
fi

nohup python main.py \
	-cc configs/default_toy_data.txt \
	--root . \
	--mode train \
	--workdir toy_work_dir/toy \
	--n_gpus_per_node 1 \
	--training_batch_size 4096 \
	--testing_batch_size 4096 \
	--sampling_batch_size 4096 \
	--master_port 6020 > log_toy.out 2> log_toy.err &
