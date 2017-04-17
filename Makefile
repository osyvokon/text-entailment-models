.PHONY: test install train glove

install:
	pip install -r requirements.txt
	python setup.py develop
	@echo
	@echo "All done"
	@echo
	@echo "Next steps:"
	@echo "  1. Consider creating a symlink in ./data/word2vec.bin pointing to pretrained word2vec vectors"
	@echo "  2. make train"
	@echo


glove: ./data/glove.6B.300d.txt

./data/glove.6B.300d.txt: ./data/glove.6B.zip
	cd data && unzip glove.6B.zip
	python -m gensim.scripts.glove2word2vec --input ./data/glove.6B.300d.txt --output ./data/word2vec.txt

./data/glove.6B.zip:
	wget http://nlp.stanford.edu/data/glove.6B.zip

train-attention: data/train.txt
	python ./src/model_global_attention.py train \
		--data ./data/train-1k.txt \
		--vocab-limit 10000 \
		./trained/model_global_attention.bin

train-conditional: data/train.txt
	python ./src/model_conditional.py train \
		--data ./data/train.txt \
		--vocab-limit 70000 \
		./trained/model_conditional.bin

train-bowman: data/train.txt
	python ./src/model_bowman.py train \
		--data ./data/train.txt \
		--vocab-limit 70000 \
		./trained/model_bowman.bin


data/snli_1.0:
	cd data && wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
	cd data && unzip snli_1.0.zip
	rm ./data/snli_1.0.zip

data/train.txt: data/snli_1.0
	python ./src/prepare_data.py ./data/snli_1.0/ ./data/

test:
	py.test --doctest-modules ./src/*.py
