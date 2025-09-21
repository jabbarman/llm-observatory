.PHONY: setup train-tiny serve-local

setup:
	@echo "Setting up environment..."
	python -m pip install -U pip setuptools wheel
	pip install -r requirements.txt

train-tiny:
	@echo "Stub: train tiny GPT model"

serve-local:
	@echo "Stub: start local server"
