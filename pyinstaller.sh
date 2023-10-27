#!/bin/bash

rm -rf dist/gputopia-worker-$1

pyinstaller $2 --name gputopia-worker-$1 --additional-hooks-dir=./hooks ai_worker/__main__.py
