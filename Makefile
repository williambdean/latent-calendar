test:
	uv run pytest

test.download:
	CI=INTEGRATION uv run pytest -k test_load_func

cov:
	uv run pytest --cov-report html
	open htmlcov/index.html

format:
	uv run pre-commit run --all-files

html:
	open http://localhost:8000/
	uv run mkdocs serve
