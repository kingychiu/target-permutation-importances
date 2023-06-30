
lint:
	isort --profile black ./target_permutation_importance
	black ./target_permutation_importance
	ruff check ./target_permutation_importance
	mypy target_permutation_importance
test:
	pytest --cov=target_permutation_importance --no-cov-on-fail --cov-fail-under=100 --cov-report=term-missing:skip-covered ./tests