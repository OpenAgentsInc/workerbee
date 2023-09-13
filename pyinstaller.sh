#!/bin/bash

pyinstaller --onefile --name gputopia-worker --additional-hooks-dir=./hooks ai_worker/__main__.py
