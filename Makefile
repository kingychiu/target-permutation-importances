
lint:
	isort --profile black ./target_permutation_importances
	black ./target_permutation_importances
	ruff check ./target_permutation_importances
	mypy target_permutation_importances
test:
	pytest --cov=target_permutation_importances --no-cov-on-fail --cov-fail-under=100 --cov-report=term-missing:skip-covered ./tests