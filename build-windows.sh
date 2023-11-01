#!/bin/bash -e

set -o xtrace

./build-bin.sh cuda win-64 "-DLLAMA_CUBLAS=1" 

./build-bin.sh opencl win-64 "-DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast" 

./upload.sh
