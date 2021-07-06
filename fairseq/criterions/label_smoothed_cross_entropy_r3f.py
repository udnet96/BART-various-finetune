# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from omegaconf import II
from sms import make_logger


@register_criterion("label_smoothed_cross_entropy_r3f")
class LabelSmoothedCrossEntropyR3FCriterion(FairseqCriterion):
    def __init__(
            self, task, sentence_avg, label_smoothing, eps,
            r3f_lambda, noise_type, ignore_prefix_size=0, report_accuracy=False
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.noise_type = noise_type
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        if self.noise_type in {"normal"}:
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == "uniform":
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        else:
            raise Exception(f"unrecognized noise type {self.noise_type}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--eps', type=float, default=1e-5,
                            help='noise eps')
        parser.add_argument('--r3f-lambda', type=float, default=0.1,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--noise-type', type=str, default='uniform',
                            choices=['normal', 'uniform'],
                            help='type of noises')
        # fmt: on

    def _get_symm_kl(self, noised_logits, input_logits):
        # print('noise , input : {}, {}'.format(noised_logits.size(), input_logits.size()))
        return (
                       F.kl_div(
                           F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                           F.softmax(input_logits, dim=-1, dtype=torch.float32),
                           size_average=None,
                           reduce=None,
                           reduction="sum",
                       )
                       + F.kl_div(
                   F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                   F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                   size_average=None,
                   reduce=None,
                   reduction="sum",
               )
               ) / noised_logits.size(0)

    def _get_symm_kl_wo_log(self, noised_logits, input_logits):
        # print('noise , input : {}, {}'.format(noised_logits.size(), input_logits.size()))
        return (
                       F.kl_div(
                           F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                           F.softmax(input_logits, dim=-1, dtype=torch.float32),
                           size_average=None,
                           reduce=None,
                           reduction="sum",
                       )
                       + F.kl_div(
                   F.softmax(input_logits, dim=-1, dtype=torch.float32),
                   F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                   size_average=None,
                   reduce=None,
                   reduction="sum",
               )
               ) / noised_logits.size(0)

    def forward(self, model, sample, reduce=True, wolog=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        token_embeddings = model.encoder.embed_tokens(sample["net_input"]["src_tokens"])
        input_logits = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, (input_logits, extra), sample, reduce=reduce)
        loss, nll_loss = self.compute_loss(model, input_logits, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if model.training:
            noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(
                token_embeddings
            )
            noised_embeddings = token_embeddings.clone() + noise
            # print('noise emb : {}'.format(noised_embeddings[0][0][:5]))
            # print('tok   emb : {}'.format(token_embeddings[0][0][:5]))
            # print()

            noised_logits, _ = model(
                **sample["net_input"], token_embeddings=noised_embeddings
            )
            if wolog:
                symm_kl = self._get_symm_kl_wo_log(noised_logits, input_logits[0])
            else:
                symm_kl = self._get_symm_kl(noised_logits, input_logits[0])

        if model.training:
            symm_kl = symm_kl * sample_size
            loss = loss + self.r3f_lambda * symm_kl
            # print('noised logits : {}, input logits : {}'.format(noised_logits[0][0][:4], input_logits[0][0][0][:4]))
            # print('eps : {}, symm_kl : {}'.format(self.eps, symm_kl))
            # print('-' * 10 + '\n')

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, input_logits, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if model.training:
            logging_output.update(
                symm_kl=utils.item(symm_kl.data) if reduce else symm_kl.data
            )

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

    def compute_loss(self, model, net_output, sample, reduce=True):
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1, 1)
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # print('lprobs, targets : {}, {}'.format(lprobs.size(), target.size()))
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.label_smoothing,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        symm_kl_sum = sum(log.get("symm_kl", 0) for log in logging_outputs)

        metrics.log_scalar("symm_kl", symm_kl_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
