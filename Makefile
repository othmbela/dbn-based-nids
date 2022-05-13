VENV=venv
BIN=$(VENV)/bin

# make it work on windows too
ifeq ($(OS), Windows_NT)
    BIN=$(VENV)/Scripts
endif


$(BIN)/activate: requirements.txt
	python3 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --upgrade -r requirements.txt


init: $(BIN)/activate
	mkdir -p ./data/processed/test
	mkdir -p ./data/processed/train
	mkdir -p ./data/processed/val
	mkdir -p ./data/raw


dataset: prepare_venv
	$(BIN)/python ./preprocessing/create_user_features.py
	$(BIN)/python ./preprocessing/create_product_features.py
	$(BIN)/python ./preprocessing/create_user_x_product_features.py
	$(BIN)/python ./preprocessing/create_datetime_features.py
	$(BIN)/python ./preprocessing/merge_features.py
	$(BIN)/python ./preprocessing/process.py


run: prepare_venv
	$(BIN)/python main.py --config ./configs/neuralNetwork.json
	$(BIN)/python main.py --config ./configs/recurrentNeuralNetwork.json


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
