# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, copy, random
import warnings
from argparse import Namespace
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.dataclass.utils import gen_parser_from_dataclass
# from fairseq.models.bart.hub_interface import BARTHubInterface  # dunno why but can't import
from fairseq.hub_utils import GeneratorHubInterface
from typing import Dict, List
from rouge import Rouge
from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from dataclasses import dataclass, field

from time import time
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
# DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
# DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
DEFAULT_ENCODER_JSON = "./gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "./gpt2_bpe/vocab.bpe"


@dataclass
class GPT2BPEConfi(FairseqDataclass):
    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )


@register_bpe("gpt2__", dataclass=GPT2BPEConfi)
class GPT2BPE_ud(object):
    def __init__(self, cfg):
        # encoder_json = file_utils.cached_path(DEFAULT_ENCODER_JSON)
        # vocab_bpe = file_utils.cached_path(DEFAULT_VOCAB_BPE)
        encoder_json = DEFAULT_ENCODER_JSON
        vocab_bpe = DEFAULT_VOCAB_BPE
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>", "<pad>"} else tok for tok in x.split()]
        )  # <pad> is added by ud

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")


class BARTHubInterface(GeneratorHubInterface):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/bart
    """

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]

    def encode(
            self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample

    def generate(
            self,
            tokenized_sentences: List[torch.LongTensor],
            *args,
            inference_step_args=None,
            **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        inference_step_args = inference_step_args or {}
        if "prefix_tokens" in inference_step_args:
            raise NotImplementedError("prefix generation not implemented for BART")
        else:
            bsz = len(tokenized_sentences)
            inference_step_args["prefix_tokens"] = tokenized_sentences[0].new_full(
                (bsz, 1), fill_value=self.task.source_dictionary.bos()
            ).to(device=self.device)
        return super().generate(
            tokenized_sentences,
            *args,
            inference_step_args=inference_step_args,
            **kwargs
        )

    def extract_features(
            self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
            self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
                                  tokens.eq(self.task.source_dictionary.eos()), :
                                  ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(
            self,
            masked_inputs: List[str],
            topk: int = 5,
            match_source_len: bool = True,
            **generate_kwargs
    ):
        masked_token = '<mask>'
        batch_tokens = []
        for masked_input in masked_inputs:
            assert masked_token in masked_input, \
                "please add one {} token for the input".format(masked_token)

            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '<s> ' + text_spans_bpe + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            ).long()
            batch_tokens.append(tokens)

        # ensure beam size is at least as big as topk
        generate_kwargs['beam'] = max(
            topk,
            generate_kwargs.get('beam', -1),
        )
        generate_kwargs['match_source_len'] = match_source_len
        batch_hypos = self.generate(batch_tokens, **generate_kwargs)

        return [
            [(self.decode(hypo['tokens']), hypo['score']) for hypo in hypos[:topk]]
            for hypos in batch_hypos
        ]


def get_lprobs_and_target(model, net_output, sample):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    target = model.get_targets(sample, net_output)
    ignore_prefix_size = 0
    if ignore_prefix_size > 0:
        if getattr(lprobs, "batch_first", False):
            lprobs = lprobs[:, ignore_prefix_size:, :].contiguous()
            target = target[:, ignore_prefix_size:].contiguous()
        else:
            lprobs = lprobs[ignore_prefix_size:, :, :].contiguous()
            target = target[ignore_prefix_size:, :].contiguous()
    return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


def rouge_reward(cfg, task, model, sample, rouge_weight=[0.05, 0.95 / 2, 0.95 / 2]):
    bpe = GPT2BPE_ud(cfg)
    net_output = model(**sample['net_input'])
    output_tokens = net_output[0]  # ( x, y, vocab size ) x * y = 1 summa size

    sentence_tok = torch.argmax(utils.log_softmax(output_tokens, dim=-1), -1)  # maxpool
    sentence_txt = bpe.decode(task.target_dictionary.string(sentence_tok))

    lprobs, target = get_lprobs_and_target(model, net_output, sample)
    ignore_index = 0
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        target_ig = target[non_pad_mask]  # ( summa size, vocab size )

    target_txt = bpe.decode(task.target_dictionary.string(target_ig)).replace('<pad>', '')
    st_time = time()
    bart = model
    hypotheses_batch = [sentence_txt]
    tgt_batch = [target_txt]
    """
    bart = BARTHubInterface(cfg, task, bart)
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'
    bart.cuda()
    bart.eval()
    bart.half()
    src_tokens = sample['net_input']['src_tokens']
    tgt_tokens = sample['target']
    src_batch, tgt_batch = [], []
    for idx, toks in enumerate(src_tokens):
        src_batch.append(bart.decode(toks).replace('<pad>', ''))
        tgt_batch.append(bart.decode(tgt_tokens[idx]).replace('<pad>', ''))

    File"/home/udnet/fairseq1/fairseq/data/encoders/gpt2_bpe.py", line
    41, in < listcomp >
    [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
    ValueError: invalid literal for int() with base 10: '<pad>'


    with torch.no_grad():
        hypotheses_batch = bart.sample(src_batch, beam=beam, lenpen=lenpen,
                                       max_len_b=max_len_b,
                                       min_len=min_len, no_repeat_ngram_size=3)
    """
    ed_time = - (st_time - time()) / 60.
    rouge_eval = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"], stats=['f'])
    R_scores = []
    for i, hypo in enumerate(hypotheses_batch):
        scores = rouge_eval.get_scores(hypotheses_batch[i], tgt_batch[i])
        # consider oracle as ref !
        R_scores.append(rouge_weight[0] * (1. - scores[0]['rouge-1']['f']) +
                        rouge_weight[1] * (1. - scores[0]['rouge-2']['f']) +
                        rouge_weight[2] * (1. - scores[0]['rouge-l']['f']))
    for i, R in enumerate(R_scores):
        # if R < 0.1:
        #    logger.info('R : {}\nsample : {}\ntarget : {}'.format(R, hypotheses_batch[i], tgt_batch[i]))
        if np.random.random() > 0.95:
            logger.info(
                '5 % data monitoring for debugging..\nR : {}\nsample : {}\ntarget : {}'.format(R, hypotheses_batch[i],
                                                                                               tgt_batch[i]))

    reward = torch.tensor(R_scores, dtype=torch.float32, device=torch.device("cuda"))
    reward.requires_grad_(False)
    # print('time : {:0.2f} min'.format(ed_time))
    return reward.contiguous().detach()


def rouge_reward_RL(cfg, task, model, sample, rouge_weight=[0.05, 0.95 / 2, 0.95 / 2]):
    bpe = GPT2BPE_ud(cfg)
    net_output = model(**sample['net_input'])
    output_tokens = net_output[0]  # ( x, y, vocab size ) x * y = 1 summa size

    sentence_tok = torch.argmax(utils.log_softmax(output_tokens, dim=-1), -1)  # maxpool
    sentence_txt = bpe.decode(task.target_dictionary.string(sentence_tok))

    lprobs, target = get_lprobs_and_target(model, net_output, sample)
    ignore_index = 0
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        target_ig = target[non_pad_mask]  # ( summa size, vocab size )

    target_txt = bpe.decode(task.target_dictionary.string(target_ig)).replace('<pad>', '')
    st_time = time()
    bart = model
    hypotheses_batch = [sentence_txt]
    tgt_batch = [target_txt]
    """
    bart = BARTHubInterface(cfg, task, bart)
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'
    bart.cuda()
    bart.eval()
    bart.half()
    src_tokens = sample['net_input']['src_tokens']
    tgt_tokens = sample['target']
    src_batch, tgt_batch = [], []
    for idx, toks in enumerate(src_tokens):
        src_batch.append(bart.decode(toks).replace('<pad>', ''))
        tgt_batch.append(bart.decode(tgt_tokens[idx]).replace('<pad>', ''))

    File"/home/udnet/fairseq1/fairseq/data/encoders/gpt2_bpe.py", line
    41, in < listcomp >
    [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
    ValueError: invalid literal for int() with base 10: '<pad>'


    with torch.no_grad():
        hypotheses_batch = bart.sample(src_batch, beam=beam, lenpen=lenpen,
                                       max_len_b=max_len_b,
                                       min_len=min_len, no_repeat_ngram_size=3)
    """
    ed_time = - (st_time - time()) / 60.
    rouge_eval = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"], stats=['f'])
    R_scores = []
    for i, hypo in enumerate(hypotheses_batch):
        scores = rouge_eval.get_scores(hypotheses_batch[i], tgt_batch[i])
        # consider oracle as ref !
        R_scores.append(rouge_weight[0] * (1. - scores[0]['rouge-1']['f']) +
                        rouge_weight[1] * (1. - scores[0]['rouge-2']['f']) +
                        rouge_weight[2] * (1. - scores[0]['rouge-l']['f']))
    for i, R in enumerate(R_scores):
        # if R < 0.1:
        #    logger.info('R : {}\nsample : {}\ntarget : {}'.format(R, hypotheses_batch[i], tgt_batch[i]))
        if np.random.random() > 0.95:
            logger.info(
                '5 % data monitoring for debugging..\nR : {}\nsample : {}\ntarget : {}'.format(R, hypotheses_batch[i],
                                                                                               tgt_batch[i]))

    reward = torch.tensor(R_scores, dtype=torch.float32, device=torch.device("cuda"))
    # reward.requires_grad_(False)
    # print('time : {:0.2f} min'.format(ed_time))
    return reward


class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        self.cfg = cfg
        self.datasets = {}
        self.dataset_to_epoch_iter = {}
        self.cnt = 0

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False, wolog=False,
            emb_model=None, is_cossim=False, bpe=None
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            # label smoothed xent
            if wolog:
                loss, sample_size, logging_output = criterion(model, sample, wolog=wolog)
            elif is_cossim:
                loss, sample_size, logging_output = criterion(model, sample, wolog=wolog,
                                                              emb_model=emb_model, is_cossim=is_cossim, bpe=bpe,
                                                              tgt_dict=self.target_dictionary, cnt=self.cnt)
                self.cnt += 1
            else:
                loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def train_step_withRL(
            self, cfg, task, sample, model, criterion, optimizer, update_num, ignore_grad=False,
            wolog=False, emb_model=None, is_cossim=False, bpe=None,
            rl_gamma=0.9984, bart=None, info_freq=40
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            # label smoothed xent, logging output : dict
            # TODO : loss_rl by rouge-1, rouge-2, rouge-L
            is_reduce = False
            loss_ml, sample_size, logging_output, nll_loss = criterion(model, sample, reduce=is_reduce, return_nll=True)
            # if reduce True, loss ml : (0) else (bsz * tgtlen, 1) shape
            # loss_ml is label smoothed nll_loss
            if bart != None:
                reward = rouge_reward_RL(cfg, task, bart, sample)
            else:
                reward = rouge_reward_RL(cfg, task, model, sample)  # (bsz) shape
            if not is_reduce:
                loss_ml = loss_ml.view(reward.size(0), loss_ml.size(0) // reward.size(0))
                # (bsz * tgtlen, 1) -> (bsz, tgtlen)
                loss_ml = loss_ml.sum(-1)  # (bsz, tgtlen) -> (bsz)

                if update_num % info_freq == 0 and update_num != 0:
                    logger.info('loss_ml, reward : {}, {}'.format(loss_ml.data, reward.data))
                loss_rl = (loss_ml * reward).sum()
                loss_ml = loss_ml.sum()
            else:
                reward = reward.sum()
                loss_rl = (loss_ml * reward).sum()
            loss = loss_ml * (1 - rl_gamma) + loss_rl * (rl_gamma)
            logging_output["loss"] = loss.data
            logging_output["nll_loss"] = loss_ml.data
            if update_num % info_freq == 0 and update_num != 0:
                logger.info(
                    'loss, rlloss, total : {:0.3f}, {:0.3f}, {:0.3f}'.format(loss_ml.data,
                                                                             loss_rl.data, loss.data))
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
            cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        return cls(cfg, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.cfg, "data", "")

    def load_dataset(
            self,
            split: str,
            combine: bool = False,
            task_cfg: FairseqDataclass = None,
            **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (FairseqDataclass): optional task configuration stored in the checkpoint that can be used
                                         to load datasets
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError("Datasets are expected to be of type FairseqDataset")
        return self.datasets[split]

    def filter_indices_by_size(
            self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practice
        # this will be ``True`` for most datasets that inherit from
        # ``FairseqDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def build_model(self, cfg: FairseqDataclass):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            cfg (FairseqDataclass): configuration object

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self)
        model = quantization_utils.quantize_model_scalar(model, cfg)
        return model

    def build_criterion(self, cfg: DictConfig):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            cfg (omegaconf.DictConfig): configration object

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(cfg, self)

    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def build_dataset_for_inference(
            self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    def build_tokenizer(self, args):
        """Build the pre-tokenizer for this task."""
        return encoders.build_tokenizer(args)

    def build_bpe(self, args):
        """Build the tokenizer for this task."""
        return encoders.build_bpe(args)


class LegacyFairseqTask(FairseqTask):
    def __init__(self, args: Namespace):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.args, "data", "")

    def build_model(self, args: Namespace):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(args, self)
        model = quantization_utils.quantize_model_scalar(model, args)
        return model

    def build_criterion(self, args: Namespace):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(args, self)
