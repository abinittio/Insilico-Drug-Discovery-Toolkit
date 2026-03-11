.PHONY: ui test lint train predict screen clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

ui: ## Launch Streamlit web UI
	streamlit run app_ui.py

test: ## Run smoke tests
	python -m pytest tests/ -v

lint: ## Run linting checks
	python -m flake8 model.py featurizer.py config.py inference.py abuse_predictor.py pharmacology_rules.py --max-line-length 120 --ignore E501,W503

train: ## Train the MAT transporter model
	python main.py --mode train --use_cache

train-herg: ## Train the hERG cardiotoxicity model
	python train_herg.py

train-cyp: ## Train the CYP450 metabolism model
	python train_cyp.py

predict: ## Predict a single molecule (usage: make predict SMILES="CCO")
	python main.py --mode predict --smiles "$(SMILES)"

screen: ## Virtual screening (usage: make screen INPUT=molecules.txt TARGET=DAT)
	python main.py --mode screen --input $(INPUT) --target $(TARGET)

validate: ## Run validation on known drugs
	python validate_stimulants.py
	python external_validation_abuse.py

clean: ## Remove cached files and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
