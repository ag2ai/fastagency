#!/usr/bin/env bash

export TERMINAL_WIDTH=80

set -e
set -x

# build docs/docs_src/getting_started
cd docs/docs_src/getting_started && \
    cookiecutter -f -o mesop --no-input https://github.com/airtai/cookiecutter-fastagency.git app_type=mesop && \
    cd mesop && tree my_fastagency_app | sed 's/^/        /' > folder_structure.txt && cd .. && \
    cookiecutter -f -o fastapi --no-input https://github.com/airtai/cookiecutter-fastagency.git app_type=fastapi+mesop && \
    cd fastapi && tree my_fastagency_app | sed 's/^/        /' > folder_structure.txt && cd .. && \
    cookiecutter -f -o nats_n_fastapi --no-input https://github.com/airtai/cookiecutter-fastagency.git app_type=nats+fastapi+mesop && \
    cd nats_n_fastapi && tree my_fastagency_app | sed 's/^/        /' > folder_structure.txt && cd .. && \
    cd ../../..


# build docs
cd docs; python docs.py build
