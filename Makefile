.PHONY: test
test:
	python -m pytest --version
	python -m pytest -sv test/


.PHONY: lint 
lint:
	pre-commit --version
	pre-commit run --all-files --show-diff-on-failure --color always


.PHONY: typecheck
typecheck:
	mypy --version
	mypy viscm


.PHONY: ci
ci: lint typecheck test
