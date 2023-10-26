# GPUtopia Workerbee


## How to use the worker:

- first set up an account at gputopia.ai, this is the easiest way to ensure your funds are swept correctly.
- for now, we only support alby logins.   i know this isn't ideal.   but it's easier for now.  in the future, any ln-wallet should work to log in and claim control over a given lnurl.
- download or build a release, stick it somewhere nice (`/usr/bin/gputopia-worker`)
- from the command-line try this:  `gputopia-worker --test_model TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M`, maybe paste the results into a <a href="https://discordapp.com/channels/1157469556768514068/1158485685867655351">discord channel</a> for fun and discussion
- if that works, run `gputopia-worker --ln_address your-ln@address-goes-here`

## Worker command line options:

```
usage: gputopia-worker [-h] [--auth_key AUTH_KEY] [--queen_url QUEEN_URL] [--ln_address LN_URL] [--loops LOOPS] [--debug]
                            [--test_model TEST_MODEL] [--test_max_tokens TEST_MAX_TOKENS] 
                            [--main_gpu MAIN_GPU] [--tensor_split TENSOR_SPLIT] [--force_layers FORCE_LAYERS]
                            [--layer_offset LAYER_OFFSET] [--version]

options:
  -h, --help                          show this help message and exit
  --version                           output version and exit
  --auth_key AUTH_KEY                 access_token for account login
  --queen_url QUEEN_URL               coordinator url (wss://queenbee.gputopia.ai/worker)
  --ln_address LN_ADDRESS             lightning address (xxxxx@getalby.com)
  --loops LOOPS                       quit after getting this number of jobs
  --debug                             verbose debugging info
  --test_model TEST_MODEL             specify a HF_REPO/PATH[:FILTER?] to test
  --test_max_tokens TEST_MAX_TOKENS   more == longer test
  --main_gpu MAIN_GPU                 default "0"
  --tensor_split TENSOR_SPLIT         default "even split", specify comma-delimited list of numbers
  --force_layers FORCE_LAYERS         default, guess layers based on model size
  --layer_offset LAYER_OFFSET         default "2" (fudge guess down by 2, leaving more room for context)
```

## How to build the worker from source:

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
- `poetry run gputopia_worker`

### Run a re-quantization on a gguf
- `poetry run quantize_gguf`



### Run tests to be sure it really works

`PYTHONPATH=. pytest tests/`


### Build your own EXE

`pyinstaller --onefile --name gputopia-worker --additional-hooks-dir=./hooks ai_worker/__main__.py`
