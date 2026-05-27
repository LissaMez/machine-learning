.PHONY: format lint test check

format:
	black bot tests

lint:
	black --check bot tests
	flake8 bot tests

test:
	pytest

check: lint test
