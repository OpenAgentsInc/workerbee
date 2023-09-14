#!/bin/bash -e

set -o xtrace

python -mvenv build-cuda

. build-cuda/scripts/activate
CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install

./pyinstaller.sh cuda-win-64
deactivate


python -mvenv build-opencl
. build-opencl/scripts/activate
CMAKE_ARGS="-DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast" FORCE_CMAKE=1 poetry install
./pyinstaller.sh opencl-win-64
deactivate

./upload.sh
