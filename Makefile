.PHONY: build_run visualise visualise_2d setup_venv create_env clean

setup: create_env setup_venv

build_run:
	go run .

visualise: setup_venv
	./venv/bin/python analysis/visualise_pareto.py checked_solutions.tsv analysis/pareto_visualisation.png

visualise_2d: setup_venv
	./venv/bin/python analysis/visualise_2d.py checked_solutions.tsv analysis/pareto_2d.png

setup_venv: venv/bin/activate

venv/bin/activate: requirements.txt
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	touch venv/bin/activate

create_env:
	@if [ ! -f .env ]; then \
		echo "TRAINING_MODE=true" > .env; \
		echo "MAX_FACILITIES=6" >> .env; \
		echo "ITERATIONS=5000" >> .env; \
		echo "POPULATION_SIZE=5" >> .env; \
		echo "CHECKED_SOLUTIONS_FILE=checked_solutions.tsv" >> .env; \
		echo ".env file created with default values"; \
	else \
		echo ".env file already exists"; \
	fi
