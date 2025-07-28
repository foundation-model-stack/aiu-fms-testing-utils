#!/bin/bash

CI=${1:-0}
PYTHON_VERSION=${2:-3.9}

if [ "$CI" -eq 1 ]; then
    set -e
fi

run_mypy() {
    echo "Running mypy on $1"
    if [ "$CI" -eq 1 ] && [ -z "$1" ]; then
        mypy --python-version "${PYTHON_VERSION}" "$@"
        return
    fi
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy aiu_fms_testing_utils
run_mypy scripts