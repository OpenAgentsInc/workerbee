#!/bin/bash -e

gpu="$1"
arch="$2"
cmake="$3"

if [ -z "$cmake" -o -z "$gpu" ]; then
    echo usage build-bin.sh gpu arch "cmake-args"
    exit 1
fi

set -o xtrace

python -mvenv "build-$gpu"

# python is absurd putting these in different places
. build-$gpu/bin/activate 2> /dev/null || . build-$gpu/scripts/activate

pip uninstall -y llama-cpp-python

# windows/linux cache rm (poetry cache control is kinda blunt/broken)
rm -f ~/AppData/Local/pypoetry/Cache/artifacts/*/*/*/*/llama*
rm -f ~/.cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="$cmake" FORCE_CMAKE=1 poetry install

./pyinstaller.sh $gpu-$arch

deactivate
