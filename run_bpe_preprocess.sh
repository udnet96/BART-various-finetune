#!/bin/bash

TASK=$1
bpe=$2

if [ $# -ne 2 ] ; then
  echo "bash run_bpe_preprocess.sh TASK bpe"
  exit 0
fi

for SPLIT in ${bpe}
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "${TASK}/${SPLIT}.${LANG}" \
    --outputs "${TASK}/${SPLIT}.bpe.${LANG}" \
    --workers 60 \
    --keep-empty;
  done
done

