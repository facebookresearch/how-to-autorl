
# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of the sweepers
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev check format pre-commit clean build clean-doc clean-build test doc

help:
	@echo "Makefile AutoRL Sweepers"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* test             to run the tests"
	@echo "* doc              to generate and view the html files"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
BLACK ?= python -m black
ISORT ?= python -m isort --profile black
PYDOCSTYLE ?= python -m pydocstyle
PRECOMMIT ?= pre-commit
FLAKE8 ?= python -m flake8

DIR := ${CURDIR}
DIST := ${CURDIR}/dist
DOCDIR := ${CURDIR}/docs
INDEX_HTML := file://${DOCDIR}/html/build/index.html

install-dev:
	$(PIP) install -e ".[dev,dehb,pbt,pb2,examples]"
	pre-commit install

check-black:
	$(BLACK)  hydra_plugins examples tests --check || :

check-isort:
	$(ISORT) hydra_plugins tests --check || :

check-pydocstyle:
	$(PYDOCSTYLE) hydra_plugins || :

check-flake8:
	$(FLAKE8) hydra_plugins || :
	$(FLAKE8) tests || :

# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: check-black check-isort check-flake8 check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) hydra_plugins examples tests

format-isort:
	$(ISORT) hydra_plugins tests

format: format-black format-isort

test:
	$(PYTEST) -v --cov=hydra_plugins tests --durations=20

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-doc clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

doc:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}