setup:
	python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

quickrun:
	bash scripts/reproduce.sh

fullrun:
	bash scripts/reproduce.sh --full

artifacts:
	bash scripts/make_artifacts.sh
