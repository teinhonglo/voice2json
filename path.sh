#!/usr/bin/env bash

export PYTHONNOUSERSITE=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Activate a Conda environment here only when needed.
# Example:
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate qwen3-slu
