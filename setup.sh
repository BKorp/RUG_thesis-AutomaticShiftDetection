#!/bin/bash

# Activate python venv
source ./env/bin/activate

# Go into ./data and run data_prep.py
cd data
python data_prep.py
cd ..

# Go into ./external_tools and run tools.sh
cd external_tools
bash tools.sh
cd ..