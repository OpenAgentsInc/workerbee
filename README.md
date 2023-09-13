local installation for llama worker

when building, do this:

### this is a CUDA/NVIDIA build
`CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install`


### for osx/metal:

`CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install`

if you want it to see the gpus!

### for CLBLAST

get this: 
https://github.com/KhronosGroup/OpenCL-SDK/releases

put it in c:/opencl-sdk


git clone https://github.com/CNugteren/CLBlast.git
mkdir CLBlast/build
cd CLBlast/build
cmake .. -DOPENCL_ROOT=C:/OpenCL-SDK -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cmake --install . --prefix C:/CLBlast

`CMAKE_ARGS="-DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast" FORCE_CMAKE=1 poetry install`

### run tests
`PYTHONPATH=. pytest tests/`
