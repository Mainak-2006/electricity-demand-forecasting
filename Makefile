# BijliPredict AI — developer task runner.
#
#   make install   # create .venv and install deps + dev tools (uv)
#   make data      # generate the synthetic household dataset
#   make train     # train the forecaster (deterministic by default)
#   make eval      # evaluate on the held-out test split
#   make predict   # 7-day forecast + 90-day projection + bill estimate
#   make report    # build the Excel + HTML reports from outputs/
#   make serve     # run the FastAPI JSON service (uvicorn)
#   make app       # run the Streamlit dashboard
#   make test      # run the pytest suite
#   make lint      # ruff check
#   make format    # ruff format
#   make all       # data -> train -> eval -> predict -> report
#
# The individual scripts still work directly; `make install` also installs the
# unified console script, e.g. `bijli predict --preset bses` or `bijli train`.

SHELL := /bin/bash

VENV ?= .venv
PY   := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

# Extra CLI args, e.g. `make train ARGS="--epochs 50"`
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help install data train eval predict report serve app test lint format all clean

help:
	@echo "Targets: install data train eval predict report serve app test lint format all clean"

install:
	uv sync --extra dev --extra serve

data:
	$(PY) generate_data.py $(ARGS)

train:
	$(PY) train.py --deterministic $(ARGS)

eval:
	$(PY) evaluate.py $(ARGS)

predict:
	$(PY) predict.py $(ARGS)

report:
	$(PY) report.py $(ARGS)

serve:
	$(PY) -m uvicorn api:app --host 0.0.0.0 --port 8000 $(ARGS)

app:
	$(PY) -m streamlit run app.py $(ARGS)

test:
	$(PY) -m pytest

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

all: data train eval predict report

clean:
	rm -rf .pytest_cache .ruff_cache
	rm -f artifacts/*.pt artifacts/*.joblib
	rm -f outputs/*
