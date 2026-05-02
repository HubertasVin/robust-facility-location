build_run:
	go run .

visualise:
	./venv/bin/python visualise_pareto.py checked_solutions.tsv pareto_visualisation.png