#!/bin/bash

PYTHON="python3.8"
PIP="pip3"
TORCH=`$PYTHON -c "import torch; print(torch.__version__)"`
CUDA=`$PYTHON -c "import torch; print(torch.version.cuda)"`

$PIP install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$PIP install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$PIP install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$PIP install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$PIP install torch-geometric