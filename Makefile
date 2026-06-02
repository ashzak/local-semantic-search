.PHONY: install install-dev install-transformers index index-hashing run test docker-build docker-up clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e '.[dev]'

install-transformers:
	python -m pip install -e '.[transformers]'

index:
	semantic-search build docs

index-hashing:
	semantic-search build docs --backend hashing

run:
	uvicorn semantic_search.app:app --reload

test:
	pytest

docker-build:
	docker compose build

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
