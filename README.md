# BART-various-finetune
BART model : [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)

BART github : [https://github.com/pytorch/fairseq/tree/master/examples/bart](https://github.com/pytorch/fairseq/tree/master/examples/bart)

## Introduction

BART is sequence-to-sequence model trained with denoising as pretraining objective. We show that this pretraining objective is more generic and show that we can match [RoBERTa](../roberta) results on SQuAD and GLUE and gain state-of-the-art results on summarization (XSum, CNN dataset), long form generative question answering (ELI5) and dialog response genration (ConvAI2). See the associated paper for more details.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
`bart.large.mnli` | `bart.large` finetuned on `MNLI` | 400M | [bart.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz)
`bart.large.cnn` | `bart.large` finetuned on `CNN-DM` | 400M | [bart.large.cnn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
`bart.large.xsum` | `bart.large` finetuned on `Xsum` | 400M | [bart.large.xsum.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)

## Results

**[CNN/Daily Mail](http://nlpprogress.com/english/summarization.html)**
_(test set, no additional data used)_

Model | R1 | R2 | RL
---|---|---|---
`BERTSUMEXTABS` | 42.13 | 19.60 | 39.18
`bart.large` | 44.16 | 21.28 | 40.90

## Example usage

##### Load BART from torch.hub (PyTorch >= 1.1):
```python
import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()  # disable dropout (or leave in train mode to finetune)
```

#### Evaluating the `bart.large.cnn` model:
- Follow instructions [here](https://github.com/abisee/cnn-dailymail) to download and process into data-files such that `test.source` and `test.target` has one line for each non-tokenized sample.
- For simpler preprocessing, you can also `wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz`, although there is no guarantee of identical scores
- `huggingface/transformers` has a simpler interface that supports [single-gpu](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/run_eval.py) and [multi-gpu](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/run_distributed_eval.py) beam search.
    In `huggingface/transformers`, the BART models' paths are `facebook/bart-large-cnn` and `facebook/bart-large-xsum`.

In `fairseq`, summaries can be generated using:

```bash
cp data-bin/cnn_dm/dict.source.txt  checkpoints/
python examples/bart/summarize.py \
  --model-dir pytorch/fairseq \
  --model-file bart.large.cnn \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo
```

For calculating rouge, install `files2rouge` from [here](https://github.com/pltrdy/files2rouge).

```bash
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Tokenize hypothesis and target files.
cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
# Expected output: (ROUGE-2 Average_F: 0.21238)
```


## Finetuning

Xsum text data : [https://github.com/EdinburghNLP/XSum](https://github.com/EdinburghNLP/XSum)

CNNDM text data : [https://github.com/abisee/cnn-dailymail](https://github.com/abisee/cnn-dailymail)

- [Finetuning on CNN-DM](README.summarization.md)
- 
Follow 1-3 to abtain processed training data 

Example normal fine-tuning on CNN-DM
```bash
TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/path/to/bart/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train cnn_dm-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

## Low-resource
```
bash run_init.sh xsum-bin_low
```

## Full-data
```
bash run_init_full.sh xsum-bin
```

To conduct another fine-tuning, you need to modify the bash file.

validation options : ROUGE_valdation, Mid-epoch vaildation
```
TRAIN_FILE=fairseq_cli/train.py
#TRAIN_FILE=fairseq_cli/train_midval_full.py
#TRAIN_FILE=fairseq_cli/train_midval.py
#TRAIN_FILE=fairseq_cli/train_Rval_midval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_Rval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_Rval_midval_cnndmlow.py
```
And also you have to match the train file with your criterion above (if you use another criterion)
```
#TRAIN_FILE=fairseq_cli/train_CS_Rval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_R3F_CS_Rstoplow_midval.py
#TRAIN_FILE=fairseq_cli/train_RRL_Rval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_RRL_Rval_midval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_rougeRL_Rval_midval.py
#TRAIN_FILE=fairseq_cli/train_R3F_cossim_Rstop.py
#TRAIN_FILE=fairseq_cli/train_R3F_cossim_Rstoplow_Sbert.py
```
Criterion : 3 options - R3F loss, Cosine-similarity(CS) loss, ROUGE_reinforcement loss (R3F + CS also available)
```
#CRITERIONN=r3f
#CRITERIONN=r3f_CS0.3
#CRITERIONN=CS0.3
#CRITERIONN=R_RL
CRITERIONN=norm

TRAIN_CRITERION=label_smoothed_cross_entropy
#TRAIN_CRITERION=label_smoothed_cross_entropy_r3f
#TRAIN_CRITERION=label_smoothed_cross_entropy_r3f_cossim
#TRAIN_CRITERION=label_smoothed_cross_entropy_cossim
#TRAIN_CRITERION=semantic_similarity_loss
```
