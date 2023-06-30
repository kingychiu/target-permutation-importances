
lint:
	isort --profile black ./target-permutation-importance
	black ./target-permutation-importance
	ruff check ./target-permutation-importance

test:
	# --p