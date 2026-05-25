.PHONY: help install download eda dashboard pdf report all validate test lint format clean

help:
	@echo "Targets:"
	@echo "  install     Install runtime + dev + report dependencies (editable)"
	@echo "  download    Fetch titanic5.csv into data/raw/"
	@echo ""
	@echo "  eda         Run the CLI EDA pipeline (console + outputs/figures/*.png)"
	@echo "  dashboard   Build the interactive HTML dashboard"
	@echo "  pdf         Build the classical EDA PDF"
	@echo "  report      Build the analyst report DOCX (and PDF if docx2pdf available)"
	@echo "  all         Run dashboard + pdf + report"
	@echo ""
	@echo "  validate    Reconcile row counts across raw -> cleaned -> engineered stages"
	@echo "  test        Run the pytest suite (49 tests)"
	@echo "  lint        Run ruff lint and format checks"
	@echo "  format      Auto-fix lint and reformat"
	@echo "  clean       Remove generated figures and caches"

install:
	pip install -e ".[dev,reports]"

download:
	python scripts/download_data.py

eda:
	python scripts/run_eda.py

dashboard:
	python dashboard/generate.py

pdf:
	python dashboard/generate_pdf.py

report:
	python reports/build_analyst_report.py

all: dashboard pdf report

validate:
	python reports/validate_dataset.py

test:
	pytest tests/ -v

lint:
	ruff check src tests scripts dashboard reports
	ruff format --check src tests scripts dashboard reports

format:
	ruff check --fix src tests scripts dashboard reports
	ruff format src tests scripts dashboard reports

clean:
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf .pytest_cache
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf dashboard/_pdf_charts/*.png
