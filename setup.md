# Setup might be a bit tricky :)

## Create conda environment
``conda create --name nemo-tts python==3.10.12``
``conda init``
now activate it

``conda activate nemo-tts``

## Packages installation

``pip install -r requirements_tts.txt``

``pip install "nemo_toolkit[all]==1.23.0"``

## Test run
Try to run our base test notebook in `test/Our-run-test.ipynb`