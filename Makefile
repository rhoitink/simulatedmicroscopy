.PHONY: docs

check: black flake test

venv:
	python -m venv venv

install:
	python -m pip install .

dev:
	python -m pip install -e .[dev,docs]

docs:
	mkdocs build

test:
	pytest tests

coverage:
	pytest --cov simulatedmicroscopy --cov-report html

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage

flake:
	flake8 simulatedmicroscopy tests

black:
	black simulatedmicroscopy tests
