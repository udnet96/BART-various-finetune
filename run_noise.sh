#!/bin/bash

TEXT=$1

if [ $# -ne 1 ] ; then
    echo "bash run_noise.sh TEXT"
    exit 0
fi
# TEXT=train_unsup_2000.hypo
python paraphrase/paraphrase.py \
  --paraphraze-fn noise_bpe \
  --word-dropout 0.2 \
  --word-blank 0.2 \
  --word-shuffle 3 \
  --data-file ${TEXT} \
  --output ${TEXT}_noise_bpe \
  --bpe-type subword