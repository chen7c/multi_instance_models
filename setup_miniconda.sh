#!/bin/bash
# Set up a clean miniconda environment for testing.
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p miniconda
export PATH=`pwd`/miniconda/bin:$PATH
conda update --yes conda
conda install numpy scipy six scikit-learn

export THEANO_FLAGS="device=cpu,floatX=float32"
