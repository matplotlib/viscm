.PHONY: test
test:
	python -m pytest --version
	python -m pytest -v test/


.PHONY: lint 
lint:
	pre-commit --version
	pre-commit run --all-files --show-diff-on-failure --color always


.PHONY: ci
ci: lint test
