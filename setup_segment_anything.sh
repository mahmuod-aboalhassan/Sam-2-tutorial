#!/bin/bash

# Clone the repository
git clone https://github.com/facebookresearch/segment-anything-2.git

# Change directory to the cloned repository and install the package
cd segment-anything-2
pip install -e .

# Change directory to the checkpoints folder and run the download script
cd checkpoints
./download_ckpts.sh
