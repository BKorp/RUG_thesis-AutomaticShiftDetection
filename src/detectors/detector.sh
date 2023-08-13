#!/bin/bash

# Set the python_environment location, this expects a venv folder
# The embfolder is the location where embeddings are located
THESENV="../../env/bin/activate"
EMBFOLDER="../../data/_embeddings/fasttext"

static () {
    # Runs the static detection pipeline
    echo "Running static for $1 to $2"
    python detector.py \
           --type static \
           --source_emb $1 \
           --target_emb $2
}

context () {
    # Runs the context detection pipeline
    echo "Running context"
    python detector.py \
           --type contextual
}

static_mt () {
    # Runs the static detection pipeline for machine translation
    echo "Running static for mt $1 to $2"
    python detector.py \
           --type static \
           --source_emb $1 \
           --target_emb $2 \
           --machine_translation
}

context_mt () {
    # Runs the context detection pipeline for machine translation
    echo "Running context for mt"
    python detector.py \
           --type contextual \
           --machine_translation
}


source $THESENV

static "$EMBFOLDER/cc.en.300.mapped_unsupervised.vec" "$EMBFOLDER/cc.nl.300.mapped_unsupervised.vec"
static "$EMBFOLDER/cc.en.300.mapped_unsup_ident.vec" "$EMBFOLDER/cc.nl.300.mapped_unsup_ident.vec"
static "$EMBFOLDER/cc.en.300.mapped_semi_supervised.vec" "$EMBFOLDER/cc.nl.300.mapped_semi_supervised.vec"
static_mt "$EMBFOLDER/cc.en.300.mapped_semi_supervised.vec" "$EMBFOLDER/cc.nl.300.mapped_semi_supervised.vec"

context
context_mt