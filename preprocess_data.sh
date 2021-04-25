DATA_DIR=R21578title
python preprocess_data.py \
	--training-file ./datasets/$DATA_DIR/training.txt \
	--validation-file ./datasets/$DATA_DIR/validation.txt \
	--test-file ./datasets/$DATA_DIR/test.txt \
	--data-output ./datasets/$DATA_DIR \
	--vocab-size 2000 \
	--split-train-val False \
	--split-num 50
sudo rm -rf ./datasets/$DATA_DIR/training.csv
sudo rm -rf ./datasets/$DATA_DIR/validation.csv
sudo rm -rf ./datasets/$DATA_DIR/test.csv