local installation for llama worker

when building, do this:

### this is a CUDA/NVIDIA build
`CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install`


### for osx/metal:

`CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install`

if you want it to see the gpus!

### run tests
`PYTHONPATH=. pytest tests/`
