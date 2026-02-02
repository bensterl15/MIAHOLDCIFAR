rm -rf ./data/train.lmdb ./data/validation.lmdb
python util/convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=data/celeba/celeba-tfr --lmdb_path=./data --split=train
python util/convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=data/celeba/celeba-tfr --lmdb_path=./data --split=validation
