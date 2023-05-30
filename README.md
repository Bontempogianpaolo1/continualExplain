# ContinualExplain
Official implementation of paper [Catastrophic Forgetting in Continual Concept Bottleneck Models](https://link.springer.com/chapter/10.1007/978-3-031-13324-4_46)

# Steps

## build Environment

conda env create -f environment.yml

conda activate continual

# Setup
download data folder from [here](https://drive.google.com/file/d/1RCcjG4FAT3Y5iDRWI7aGkwSwYr0P-rxv/view?usp=sharing)


# How to prepare data
download from [Here](http://www.vision.caltech.edu/visipedia/CUB-200.html)
Then run data_processing.py

unzip data.zip
pip install pytorch-lightning
pip install avalanche-lib

## training

python training.py

## statistics

python compute_statistics.py



# How to start experiments

python main.py changing default parameters


