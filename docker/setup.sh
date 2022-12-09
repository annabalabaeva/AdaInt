export FORCE_CUDA="1"
python3.7 --version
python3.7 -m pip install -r requirements.txt
python3.7 -m pip install -v -e .
#python3.7 adaint/ailut_transform/setup.py install
python3.7 -m pip install adaint/ailut_transform/ailut-1.5.0-cp37-cp37m-linux_x86_64.whl
