PY ?= python
VENV_PY = .venv/Scripts/python

.PHONY: all venv data validate expected covariances ml backtest sanity

all: data expected covariances backtest

venv:
	$(PY) -m venv .venv
	$(VENV_PY) -m pip install -r requirements.txt

data:
	$(PY) get_data.py

validate:
	$(PY) validate_data.py

expected:
	$(PY) returns/generate_expected_returns.py

covariances:
	$(PY) risk/generate_covariances.py

ml:
	$(PY) optimization/predict_returns.py

backtest:
	$(PY) backtest/backtest.py

sanity:
	$(PY) allocation/allocation_sanity_check.py
	$(PY) risk/risk_sanity_check.py
	$(PY) returns/returns_sanity_check.py
