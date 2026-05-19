.PHONY: install download eda test clean

install:
	pip install -r requirements.txt

download:
	python scripts/download_data.py

eda:
	python scripts/run_eda.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf .pytest_cache
	rm -rf __pycache__ */__pycache__
