#!/bin/bash

pyinstaller $2 --name gputopia-worker-$1 --additional-hooks-dir=./hooks ai_worker/__main__.py
