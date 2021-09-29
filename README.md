# This repository

This is the Github repository of the paper:
Lorenzo Cian, Giuseppe Lancioni, Lei Zhang, Mirco Ianese, Nicolas Novelli, Giuseppe Serra, Francesco Maresca, *Atomistic Graph Neural Networks for metals: Application to bcc iron*.  
The repository contains the code and the data necessary to reproduce the results presented in the above paper.  
All code is available in the IPYNB format to allow for easy execution on cloud computing platforms like Google Colab. 

## Instructions

- Clone this repository and place it somewhere on your computer or on Google Drive;
- depending on whether or not you are using Colab, edit the second code cell in both notebooks according to the instructions.

The data required to reproduce the results in the paper is included in this repository.  
The two pretrained models are also included.

If you want to train your own model, place .xyz files in the data/training folder.  
If you want to reproduce the training as described by the paper, you should do so by using the following dataset: Daniele Dragoni, Tom Daff, Gabor Csanyi, Nicola Marzari, *Gaussian Approximation Potentials for iron from extended first-principles database (Data Download)*, Materials Cloud Archive 2017.0006/v2 (2017), doi: 10.24435/materialscloud:2017.0006/v2  
which is currently available [here](https://archive.materialscloud.org/2017.0006/v2).