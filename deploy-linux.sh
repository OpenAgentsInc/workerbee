#!/bin/bash -e

set -o xtrace

python -mvenv build-cuda
. build-cuda/bin/activate

pip uninstall -y llama-cpp-python
rm -f ~/.cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install
./pyinstaller.sh cuda-linux-64

deactivate


python -mvenv build-opencl
. build-opencl/bin/activate

pip uninstall -y llama-cpp-python
rm -f ~/.cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="-DLLAMA_CLBLAST=ON" FORCE_CMAKE=1 poetry install
./pyinstaller.sh opencl-linux-64

deactivate

./upload.sh
