#!/bin/bash

# this can't successfully be added to poetry, poetry has no way of specifying an arbitrary environment
pip install --force --no-deps torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
