# Training-on-Multiple-GPUs
This repository will demonstrate how to train a DNN model on multiple GPUs using PyTorch framework

## Dependencies

Download the dataset from [kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092)

Install the required dependencies using the command `pip install -r requirements.txt`

### Install Apex 
If using pip use this [link](https://github.com/NVIDIA/apex) else if conda use this [link](https://anaconda.org/conda-forge/nvidia-apex)

## Test the program

Use the below commands to run the program

`cd src/`

Replace line 35 with your ip address in train.py 

Replace lines 46, 47 and 48 with your file paths in train.py

Run the program using the command

`python train.py`

If you want to use multiple GPUs to train, use the argument `-g x` where x is the number of GPUs

For example use the command `python train.py -g 2` to use 2 GPUs

Try to change the port number in line 36 of `train.py` if you encounter Runtime Error: Address already in use
