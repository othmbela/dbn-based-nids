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


dataset: init
	$(BIN)/python ./preprocessing/cicids2017.py


run: init
	$(BIN)/python main.py --config ./configs/deepBeliefNetwork.json
	$(BIN)/python main.py --config ./configs/multilayerPerceptron.json


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
