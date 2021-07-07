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

- [Finetuning on CNN-DM](README.summarization.md)

