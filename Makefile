PYTHON := ./venv/bin/python

.PHONY: help mobility features fca visualize pipeline analysis clean clean-all

help:
	@echo "Available targets:"
	@echo "  make mobility    - Process raw mobility data"
	@echo "  make features    - Build the FCA feature matrix"
	@echo "  make fca         - Run formal concept analysis"
	@echo "  make visualize   - Generate charts and reports"
	@echo "  make pipeline    - Run the full pipeline"
	@echo "  make analysis    - Run FCA and visualization only"
	@echo "  make clean       - Remove generated files under results/"
	@echo "  make clean-all   - Remove generated files under results/ and data/processed/"

mobility:
	$(PYTHON) scripts/collect_mobility_data.py

features:
	$(PYTHON) scripts/preprocess_and_features.py

fca:
	$(PYTHON) scripts/fca_analysis.py

visualize:
	$(PYTHON) scripts/visualize_results.py

pipeline:
	cd scripts && printf '2\n' | ../venv/bin/python run_full_pipeline.py

analysis:
	$(PYTHON) scripts/fca_analysis.py
	$(PYTHON) scripts/visualize_results.py

clean:
	rm -f results/fca/*
	rm -f results/visualizations/*
	rm -f results/summary_report.txt

clean-all: clean
	rm -f data/processed/*
