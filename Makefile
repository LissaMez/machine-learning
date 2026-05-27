.PHONY: format lint test check

format:
	black bot tests
	@if [ -d original_implementations ]; then nbqa black original_implementations; fi

lint:
	black --check bot tests
	flake8 bot tests
	@if [ -d original_implementations ]; then nbqa flake8 original_implementations; fi

test:
	pytest

check: lint test
