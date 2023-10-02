# How to use the worker:

- first set up an account at gputopia.ai, this is the easiest way to ensure your funds are swept correctly.
- for now, we only support alby logins.   i know this isn't ideal.   but it's easier for now.  in the future, any ln-wallet should work to log in and claim control over a given lnurl.
- download or build a release, stick it somewhere nice (`/usr/bin/gputopia-worker`)
- from the command-line try this:  `gputopia-worker --test_model TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M`, maybe paste the results into a <a href="https://discordapp.com/channels/1157469556768514068/1158485685867655351">discord channel</a> for fun and discussion
- if that works, run `gputopia-worker --ln_url your-ln-url-goes-here`


# How to build the worker from source:

When building, please ensure you have CUDA installd or OPENCL (for AMD chips).   You can also do a METAL build for OSX.

### CUDA/NVIDIA build
`CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 poetry install`

### OSX/METAL build:

`CMAKE_ARGS="-DLLAMA_METAL=1" FORCE_CMAKE=1 poetry install`

if you want it to see the gpus!

### CLBLAST build:

get (or build) this:

https://github.com/KhronosGroup/OpenCL-SDK/releases

put it in c:/opencl-sdk or (on linux) cmake --install it

```
git clone https://github.com/CNugteren/CLBlast.git
mkdir CLBlast/build
cd CLBlast/build
cmake .. -DOPENCL_ROOT=C:/OpenCL-SDK -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cmake --install . --prefix C:/CLBlast
```

`CMAKE_ARGS="-DLLAMA_CLBLAST=ON -DCMAKE_PREFIX_PATH=C:/CLBlast/lib/cmake/CLBlast" FORCE_CMAKE=1 poetry install`


### Run a dev-mode worker
- `poetry shell`
- `poetry run gputopia_worker`


### Run tests to be sure it really works

`PYTHONPATH=. pytest tests/`


### Build your own EXE

`pyinstaller --onefile --name gputopia-worker --additional-hooks-dir=./hooks ai_worker/__main__.py`

