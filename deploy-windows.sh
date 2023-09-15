#!/bin/bash -e

set -o xtrace

python -mvenv build-cuda

. build-cuda/scripts/activate
pip uninstall -y llama-cpp-python
rm -f ~/AppData/Local/pypoetry/Cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install
./pyinstaller.sh cuda-win-64

deactivate


python -mvenv build-opencl
. build-opencl/scripts/activate

pip uninstall -y llama-cpp-python
rm -f ~/AppData/Local/pypoetry/Cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="-DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast" FORCE_CMAKE=1 poetry install
./pyinstaller.sh opencl-win-64

deactivate

./upload.sh
