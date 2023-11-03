#!/bin/bash -e

# pyenv essentials
curl https://pyenv.run | bash

sudo apt-get install -y curl git-core gcc make zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev
sudo apt-get install -y build-essential libncursesw5-dev libgdbm-dev libc6-dev tk-dev
sudo apt-get install -y libffi-dev liblzma-dev

cat > ~/.bash_pyenv << EOF
## pyenv configs
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
EOF

. ~/.bash_pyenv

pyenv install 3.11
pyenv global 3.11

# check it
python --version | grep 3.11

# install poetry to system so packages that depend on it work well
python -mpip install --upgrade pip
python -mpip install poetry
