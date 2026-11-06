.PHONY: install uninstall models test eval

install:
	bash ./install.sh

uninstall:
	bash ./uninstall.sh

models:
	uv run burnish-models pull --all

test:
	uv run pytest

eval:
	uv run python eval/run_eval.py
