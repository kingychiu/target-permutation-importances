
lint:
	isort --profile black ./target_permutation_importance
	black ./target_permutation_importance
	ruff check ./target_permutation_importance
	mypy target_permutation_importance
test:
	# --p