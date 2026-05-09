.PHONY: build_run visualise visualise_2d venv/bin/activate create_env clean

setup: create_env venv/bin/activate

build_run:
	go run .

run_experiment:
	./venv/bin/python analysis/experiments/run_experiments.py
	./venv/bin/python analysis/metrics/analyse_metrics.py
	$(MAKE) visualise_metrics

run_iqm_experiment:
	./venv/bin/python analysis/experiments/run_iqm_experiments.py
	$(MAKE) visualise_iqm_history

visualise_metrics: setup_venv
	./venv/bin/python analysis/metrics/visualise_metrics.py

visualise_pareto: setup_venv
	./venv/bin/python analysis/pareto/visualise_pareto.py checked_solutions.tsv analysis/pareto_visualisation.png

visualise_pareto_2d: 
	./venv/bin/python analysis/pareto/visualise_2d.py checked_solutions.tsv analysis/pareto_2d.png

visualise_iqm_history: setup_venv
	./venv/bin/python analysis/iqm/visualise_iqm_history.py

setup_venv: requirements.txt
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	touch venv/bin/activate

create_env:
	@if [ ! -f .env ]; then \
		cat >> .env <<-EOF
		TRAINING_MODE=false
		MAX_FACILITIES=5
		ITERATIONS=20000
		POPULATION_SIZE=20
		# TWO_D_MODE=true
		CHECKED_SOLUTIONS_FILE=checked_solutions.tsv
	EOF
	else \
		echo ".env file already exists"; \
	fi