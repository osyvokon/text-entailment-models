.PHONY: test


train: data/train.txt
	python ./src/model_conditional.py

data/snli_1.0:
	curl https://nlp.stanford.edu/projects/snli/snli_1.0.zip | tar -xzf- -C ./data/

data/train.txt:
	python ./src/prepare_data.py ./data/snli_1.0/ ./data/

test:
	py.test --doctest-modules ./src/*.py
