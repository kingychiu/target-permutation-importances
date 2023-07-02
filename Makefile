
lint:
	poetry run isort --profile black ./target_permutation_importances ./benchmarks ./tests
	poetry run black ./target_permutation_importances ./benchmarks ./tests
	poetry run ruff check ./target_permutation_importances ./benchmarks ./tests
	poetry run mypy target_permutation_importances
test:
	poetry run pytest --cov=target_permutation_importances --no-cov-on-fail --cov-fail-under=100 --cov-report=term-missing:skip-covered ./tests

run_tabular_benchmark:
	poetry run python -m benchmarks.run_tabular_benchmark

build:
	poetry build

publish:
	poetry publish