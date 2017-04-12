.PHONY: test install train

install:
	pip install -r requirements.txt
	python setup.py develop
	echo
	echo "All done"
	echo
	echo "Next steps:"
	echo "  1. Consider creating a symlink in ./data/word2vec.bin pointing to pretrained word2vec vectors"
	echo "  2. make train"
	echo



train: data/train.txt
	python ./src/model_conditional.py train \
		--data ./data/train.txt \
		--vocab-limit 70000 \
		./data/model_conditional.bin


data/snli_1.0:
	curl https://nlp.stanford.edu/projects/snli/snli_1.0.zip | tar -xzf- -C ./data/

data/train.txt:
	python ./src/prepare_data.py ./data/snli_1.0/ ./data/

test:
	py.test --doctest-modules ./src/*.py
