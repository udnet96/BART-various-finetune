# Copyright (c) Facebook, Inc. and its affiliates.
#
# Modified by SemSim authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from fairseq.data import encoders
from fairseq.data.encoders.gpt2_bpe import GPT2BPE

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics, utils

from semsim.rewarder import Rewarder
from dataclasses import dataclass, field
import torch
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from pytorch_transformers import *
from sms import make_logger

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


def sesim_loss(lprobs, target, epsilon, task=None, bpe=None, rewarder=None, output_tokens=None, ignore_index=None,
               reduce=True, loss_weight=None, debug=False):
    if loss_weight is None:
        loss_weight = 100
    ## semantic sim_loss
    sentence_tok = torch.argmax(utils.log_softmax(output_tokens, dim=-1), -1)  # maxpool
    sentence_txt = bpe.decode(task.target_dictionary.string(sentence_tok))

    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        target_ig = target[non_pad_mask]

    target_txt = bpe.decode(task.target_dictionary.string(target_ig))

    semsim_score = rewarder(target_txt, sentence_txt)
    if debug:
        print("\n\n## sentence_txt: ", sentence_txt, "\n## target_txt: ", target_txt, "\n## Reward :", semsim_score)

    # original label_smoothed_nll_loss
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss

    if debug:
        print("nll_loss, smooth_loss: ", nll_loss, smooth_loss)
        print("normal_loss, reward: ", loss, semsim_score)
    loss = loss - loss_weight * semsim_score
    # LOG : loss
    # was 1:1, increased to 1: 100 | 20191212
    # original : loss + 100*semsim_score, neg : loss - 100*semsim_score | 20191212
    if debug:
        print("===" * 10)
    return loss, nll_loss, semsim_score  # semsim_score : semsim_score


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )


@register_criterion(
    "semantic_similarity_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class SemanticSimilarityCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            cfg=None
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.task = task
        self.debugCount = 0
        # args.bpe = 'gpt2'
        self.bpe = GPT2BPE(cfg)
        """
        if args.rewarderpath == None:
            args.rewarderpath = "./semsim/trained_models/" + args.restore_file.split('/')[-1] # TODO : refactoring required
            print("args.rewarderpath not set : use %s instead."%args.rewarderpath) """
        # args.rewarderpath = "./semsim/trained_models/sample.model"  # TODO
        self.rewarder = Rewarder(cos_sim_scoring=True)
        self.loss_weight = 100

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'])
        loss, nll_loss, semsim_score = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            'semsim_score': utils.item(semsim_score) if reduce else semsim_score,  # semsim_score : int
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    # orig label smoothed xent loss
    def compute_loss2(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    # SWEM
    def compute_loss(self, model, net_output, sample, reduce=True):
        debug = False
        self.debugCount += 1
        if self.debugCount % 100 == 1:
            debug = True
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss, semsim_score = sesim_loss(
            lprobs, target, self.eps, task=self.task, bpe=self.bpe, rewarder=self.rewarder,
            output_tokens=net_output[0],
            ignore_index=self.padding_idx, reduce=reduce, loss_weight=self.loss_weight,
            debug=debug
        )

        return loss, nll_loss, semsim_score

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
                2) if ntokens > 0 else 0.,
            'semsim_score': sum(log.get('semsim_score', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
