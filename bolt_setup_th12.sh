python -m pip install --upgrade pip
pip install gpustat torchinfo

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install -r requirements.txt
pip install tinycudann/bindings/torch

# install cuda 
pip install models/csrc

# just for committing code
git config --global user.email "jiatao@apple.com"
git config --global user.name  "Jiatao Gu"
