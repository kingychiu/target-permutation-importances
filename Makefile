install_cuml:
	# https://developer.nvidia.com/cuda-12-0-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
	# https://docs.rapids.ai/install#rapids-release-selector
	poetry run pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com

lint:
	poetry run isort --profile black ./target_permutation_importances ./benchmarks ./tests
	poetry run black ./target_permutation_importances ./benchmarks ./tests
	poetry run ruff check ./target_permutation_importances ./benchmarks ./tests
	poetry run mypy target_permutation_importances

test:
	poetry run pytest --ignore=./tests/test_compute_cuml.py --cov=target_permutation_importances \
	--no-cov-on-fail --cov-fail-under=100 --cov-report=term-missing:skip-covered \
	-n auto ./tests

test_cuml:
	poetry run pytest --cov=target_permutation_importances --no-cov-on-fail --cov-report=term-missing:skip-covered ./tests/test_compute_cuml.py

run_tabular_benchmark:
	poetry run python -m benchmarks.run_tabular_benchmark

process_run_tabular_benchmark:
	poetry run python -m benchmarks.process_result_csv


build:
	rm -rf dist
	poetry build

publish_test:
	poetry run twine upload --repository testpypi dist/*

publish:
	poetry publish

doc:
	cp README.md docs/index.md
	cp benchmarks/results.md docs/benchmarks.md
	poetry run mkdocs serve