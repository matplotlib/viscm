.PHONY: lint 
lint:
	pre-commit run --all-files --show-diff-on-failure --color always
