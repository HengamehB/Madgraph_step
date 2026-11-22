PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || printf '%s' python)
# Omitting --config lets the CLI use .config when present and built-in defaults
# on a fresh clone. An explicitly supplied CONFIG remains mandatory.
CONFIG ?=
SELECTED ?= all
INPUT ?=
PARTICLES ?= input/particles.txt

.PHONY: help init setup validate run check configure-tools inputs ufo channels decays histories rates explicit-decays test
help:
	@echo 'Activate Python 3.11; edit input/particles.txt; make init; edit input/my_model.txt; make run.'
	@echo 'INPUT=path/to/model.txt overrides [project] model in CONFIG.'
	@echo 'make init reads PARTICLES=input/particles.txt; optional INPUT=path/my_model.txt sets its destination.'
	@echo 'Targets: init setup validate run check configure-tools inputs ufo channels decays histories rates explicit-decays test'
	@echo 'Example: make rates SELECTED=C0001,C0002'
	@echo 'make test runs unit tests and fast benchmark checks without Mathematica or MadGraph.'

init:
	"$(PYTHON)" pipeline_cli.py init "$(PARTICLES)" $(if $(INPUT),--model "$(INPUT)")

check configure-tools:
	"$(PYTHON)" pipeline_cli.py $@ $(if $(INPUT),"$(INPUT)") $(if $(CONFIG),--config "$(CONFIG)")

ufo channels decays histories explicit-decays:
	"$(PYTHON)" pipeline_cli.py $@ $(if $(INPUT),"$(INPUT)") $(if $(CONFIG),--config "$(CONFIG)")

setup validate run:
	"$(PYTHON)" pipeline_cli.py $@ $(if $(INPUT),"$(INPUT)") $(if $(CONFIG),--config "$(CONFIG)") --selected "$(SELECTED)"

inputs:
	@"$(PYTHON)" pipeline_cli.py inputs $(if $(INPUT),"$(INPUT)") $(if $(CONFIG),--config "$(CONFIG)")

rates:
	"$(PYTHON)" pipeline_cli.py rates $(if $(INPUT),"$(INPUT)") $(if $(CONFIG),--config "$(CONFIG)") --selected "$(SELECTED)"

test:
	"$(PYTHON)" -m unittest discover -s tests -p 'test_*.py'
	"$(PYTHON)" ci/run_benchmark.py --mode fast
