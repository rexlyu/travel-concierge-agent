.PHONY: install lint test eval eval-live adk-web adk-run tf-fmt tf-validate docker-build clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check travel_concierge tests evals

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v

eval:
	PYTHONPATH=. $(PYTHON) evals/run_evals.py

eval-live:
	PYTHONPATH=. $(PYTHON) evals/run_evals.py --adk-live

adk-web:
	adk web .

adk-run:
	adk run travel_concierge

tf-fmt:
	terraform -chdir=terraform fmt -check

tf-validate:
	terraform -chdir=terraform init -backend=false
	terraform -chdir=terraform validate

docker-build:
	docker build -t travel-concierge-agent:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
