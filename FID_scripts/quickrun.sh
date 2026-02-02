./clear.sh
sudo rm -r work_dir/cifar/*
sudo rm -r work_dir/cifar_seed_0/eval/
./build.sh
./run.sh
./logs.sh
