#!/bin/bash

TASK=$1
train_data=$2
dest=$3

if [ $# -ne 3 ] ; then
	echo "bash run_binarize.sh TASK train_data dest"
	exit 0
fi

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/${train_data}.bpe" \
  --validpref "${TASK}/val_low.bpe" \
  --destdir "${TASK}-bin_${dest}/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

