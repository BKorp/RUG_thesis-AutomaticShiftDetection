#!/bin/bash

source ../env/bin/activate

function laser_prep {
    export LASER=`realpath "./LASER"`
    cd "./LASER"
        bash ./nllb/download_models.sh ace_Latn
        bash install_external_tools.sh
        sed -i 's/model_dir=""/model_dir=$LASER/g' ./tasks/embed/embed.sh
        sed -i 's/over_write=False/over_write=True/g' ./source/embed.py
    cd ..
}


#####################################
### Cross-lingual static embeddings
#####################################
# VECALIGN - Framework to learn cross-lingual word embedding mappings
git clone https://github.com/thompsonb/vecalign.git

##############################
### Sentence alignment tools
##############################
# VECMAP - An accurate sentence alignment algorithm which is fast even for very long documents
git clone https://github.com/artetxem/vecmap.git

# LASER - A library to calculate and use multilingual sentence embeddings
git clone https://github.com/facebookresearch/LASER.git
laser_prep

##############################
### Word alignment tools
##############################
# AWESOMEALIGN - Used by van der Heden
git clone https://github.com/neulab/awesome-align.git

# AWESOMEALIGN_ASTRED - Used in ASTRED, sane defaults that should work well for our word pair comparisons
# Installed as a pip package for requirements.txt

#####################################
### Syntactic alignment tools
#####################################
# ASTRED - Tool for syntactic comparisons of sentences
git clone https://github.com/BramVanroy/astred.git
# Also installed as a pip package for requirements.txt
