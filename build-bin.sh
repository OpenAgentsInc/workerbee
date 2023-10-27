#!/bin/bash -e

gpu="$1"
arch="$2"
cmake="$3"
opts="--onefile"

if [ -z "$cmake" -o -z "$gpu" ]; then
    echo usage build-bin.sh gpu arch "cmake-args"
    exit 1
fi

with_torch=""
if [ "$gpu" == "cuda-torch" ]; then
    with_torch="--with torch"
    opts=""
fi


set -o xtrace

python -mvenv "build-$gpu"

# python is absurd putting these in different places
. build-$gpu/bin/activate 2> /dev/null || . build-$gpu/scripts/activate

pip uninstall -y llama-cpp-python

# windows/linux cache rm (poetry cache control is kinda blunt/broken)
rm -f ~/AppData/Local/pypoetry/Cache/artifacts/*/*/*/*/llama*
rm -f ~/.cache/pypoetry/artifacts/*/*/*/*/llama*

CMAKE_ARGS="$cmake" FORCE_CMAKE=1 poetry install $with_torch

python build-version.py

./pyinstaller.sh $gpu-$arch $opts

if [ "$gpu" == "cuda-torch" ]; then
    pushd dist
    tar cvf - gputopia-worker-$gpu-$arch/ | pigz -9 - > gputopia-worker-$gpu-$arch.tar.gz
    popd
fi

deactivate
