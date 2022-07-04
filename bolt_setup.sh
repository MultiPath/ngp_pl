source /coreflow/venv/bin/activate

echo "export PATH=/usr/local/cuda/bin/:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "alias blobby='aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300'"  >> ~/.bashrc
echo "source /coreflow/venv/bin/activate" >> ~/.bashrc
source ~/.bashrc

python -m pip install --upgrade pip
pip install gpustat torchinfo
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install -r requirements.txt

# just for committing code
git config --global user.email "jiatao@apple.com"
git config --global user.name  "Jiatao Gu"