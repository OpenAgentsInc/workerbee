local installation for llama worker

when building, do this:

CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install

if you want it to see the gpus!
