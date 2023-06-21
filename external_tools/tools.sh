#!/bin/bash

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
