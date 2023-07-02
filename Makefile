
lint:
	isort --profile black ./target_permutation_importances ./benchmarks ./tests
	black ./target_permutation_importances ./benchmarks ./tests
	ruff check ./target_permutation_importances ./benchmarks ./tests
	mypy target_permutation_importances
test:
	pytest --cov=target_permutation_importances --no-cov-on-fail --cov-fail-under=100 --cov-report=term-missing:skip-covered ./tests

run_tabular_benchmark:
	python -m benchmarks.run_tabular_benchmark