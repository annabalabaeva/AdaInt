conda create -n adaint python=3.7.10
conda activate adaint
conda install -c pytorch pytorch=1.8.1 torchvision cudatoolkit=10.2 -y
pip install -r requirements.txt
pip install -v -e .
python adaint/ailut_transform/setup.py install

