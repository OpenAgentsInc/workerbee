#!/bin/bash -e

set -o xtrace

./build-bin.sh cuda mac-64 "-DLLAMA_CUBLAS=1" 

./build-bin.sh opencl mac-64 "-DLLAMA_CLBLAST=ON" 

./upload.sh
