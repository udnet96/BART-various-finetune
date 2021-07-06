#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os, copy
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from sms import mes, make_logger
from fairseq.models.bart.hub_interface import BARTHubInterface
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from dataclasses import dataclass, field

rootlogger = make_logger()
logger = logging.getLogger("fairseq_cli.train")
DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


@dataclass
class GPT2BPEConfi(FairseqDataclass):
    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )


@register_bpe("gpt2_ud2", dataclass=GPT2BPEConfi)
class GPT2BPE_ud2(object):
    def __init__(self, cfg):
        encoder_json = file_utils.cached_path(DEFAULT_ENCODER_JSON)
        vocab_bpe = file_utils.cached_path(DEFAULT_VOCAB_BPE)
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>", "<pad>"} else tok for tok in x.split()]
        )  # <pad> is added by ud

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")


def main(cfg: DictConfig, args_ud=None) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    utils.import_user_module(cfg.common)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    # logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # TODO : val loss -> val rouge

    """
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    """

    """
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )
    """
    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    orig_state_dict = trainer.get_model().state_dict()

    is_cossim = True
    if is_cossim:
        #emb_model = SentenceTransformer('stsb-roberta-large')
        emb_model = SentenceTransformer('stsb-distilbert-base')
        bpe_ud = GPT2BPE_ud2(cfg)
    else:
        emb_model, bpe_ud = None, None

    is_val_by_rouge = False
    is_rouge_rl = False
    is_wo_log = False
    cur_patience = 0
    best_val_loss = 10e10
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, val_by_rouge=is_val_by_rouge,
                                          orig_state=orig_state_dict, is_rouge_rl=is_rouge_rl,
                                          is_wolog=is_wo_log, is_cossim=is_cossim, emb_model=emb_model, bpe=bpe_ud)
        logger.info('prev val best : {}'.format(best_val_loss))
        if valid_losses[0] < best_val_loss:
            best_val_loss = valid_losses[0]
            cur_patience = 0
        else:
            cur_patience += 1
        logger.info('current patience : {}'.format(cur_patience))
        logger.info('current val loss : {}'.format(valid_losses[0]))
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        logger.info('current lr : {}'.format(lr))

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    if distributed_utils.is_master(cfg.distributed_training):
        logger.info(
            "done training with {:.2f} val loss in {:.1f} mins".format(best_val_loss, float(train_meter.sum) / 60.0))
        savedir_ = ''
        for i, arg in enumerate(sys.argv):
            if str(arg).count('save-dir') > 0:
                savedir_ = str(sys.argv[i + 1]).split('/')
                savedir_ = '/'.join(savedir_[-2:])
                break
        mes("done training. | {:.2f} val loss | {} epochs | savedir : {} | {:.1f} mins".format(best_val_loss,
                                                                                               int(
                                                                                                   epoch_itr.next_epoch_idx - 1),
                                                                                               savedir_,
                                                                                               float(
                                                                                                   train_meter.sum) / 60.0))


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, val_by_rouge=False, orig_state=None,
        is_rouge_rl=False, is_wolog=False, emb_model=None, is_cossim=False, bpe=None
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))
    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):  # len(samples) == update freq. // not related to bsz..
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            if is_wolog:
                log_output = trainer.train_step(samples, rouge_rl=is_rouge_rl, wolog=is_wolog,
                                                emb_model=emb_model, is_cossim=is_cossim, bpe=bpe)
            else:
                log_output = trainer.train_step(samples, rouge_rl=is_rouge_rl,
                                                emb_model=emb_model, is_cossim=is_cossim, bpe=bpe)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        val_by_rouge = val_by_rouge
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch, rouge_val=val_by_rouge, orig_state=orig_state
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def range_devide(cfg: DictConfig, div_len=0) -> range:
    divide = div_len // cfg.distributed_training.distributed_world_size
    if divide < 0:
        print('# data < # divice')
        return range(div_len)
    elif cfg.distributed_training.distributed_world_size == 1:
        return range(div_len)
    range_start = cfg.distributed_training.distributed_rank * divide
    if int(cfg.distributed_training.distributed_rank) == int(cfg.distributed_training.distributed_world_size) - 1:
        range_end = div_len
    else:
        range_end = range_start + divide
    return range(range_start, range_end)


from fairseq.models.bart import BARTModel
import time, random
from datetime import datetime
from rouge import Rouge
from glob import glob


def summa_gen(cfg: DictConfig, target_src=None, bart=None) -> str:
    src = target_src
    # try:
    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)
    save_dir = os.path.join(cfg.checkpoint.save_dir, 'val_dir')

    start_time = time.time()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = os.path.join(save_dir, 'hypofor.val')
    filename += str(cfg.distributed_training.distributed_rank)

    mode = 'xsum'
    if mode == 'cnn':
        beam, lenpen, max_len_b, min_len, data = 4, 2.0, 140, 55, 'cnndm'
    elif mode == 'gigaword':
        beam, lenpen, max_len_b, min_len, data = 6, 1.0, 20, 0, 'gigaword'
    else:
        beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'

    if distributed_utils.is_master(cfg.distributed_training):
        print('[{}]\nfile | [{}] gen start.. \nsrc | [{}]'.format(datetime.now(),
                                                                  filename,
                                                                  src))
        print('mode | {}, beam | {}, lenpen | {}\nmax_len | {}, min_len | {}'.
              format(data, beam, lenpen, max_len_b, min_len))
    with open(src, 'r', encoding='utf8') as source:
        src_lines = source.readlines()
    rnge = range_devide(cfg, len(src_lines))

    # model gen summary write
    count = 0
    bsz = 24  # for 12GB GPU memory
    with open(filename, 'w', encoding='utf8') as fout:
        slines = []
        for idx in rnge:
            sline = src_lines[idx].strip()
            if count % bsz == 0 and count != 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen,
                                                   max_len_b=max_len_b,
                                                   min_len=min_len, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len,
                                           no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()

    end_time = time.time()
    fintime = (end_time - start_time) / 60.0
    print('finished {} in {} min'.format(os.path.basename(filename), fintime))
    del hypotheses_batch, slines
    torch.cuda.empty_cache()
    if distributed_utils.is_master(cfg.distributed_training):
        if cfg.distributed_training.distributed_world_size > 1:
            devices = [i for i in range(1, cfg.distributed_training.distributed_world_size)]
            conds = [False for i in range(1, cfg.distributed_training.distributed_world_size)]
            done_files = []
            for i in devices:
                dfile = os.path.join(save_dir, 'valgendone{}'.format(i))
                done_files.append(dfile)
            while True:
                for idx, file in enumerate(done_files):
                    if os.path.exists(file):
                        conds[idx] = True
                if all(conds):
                    break
            for df in done_files:
                if os.path.exists(df):
                    os.remove(df)
                    print('removed : {}..'.format(df))
        a = sorted(glob(filename[:-2] + '*'))
        with open(filename[:-1], 'w', encoding='utf8') as fout:
            for j in a:
                if j != filename[:-1]:
                    with open(j) as fin:
                        contents = fin.read().strip()
                    fout.write(contents.strip() + '\n')
    else:
        with open(os.path.join(save_dir, 'valgendone{}'.format(cfg.distributed_training.distributed_rank)), 'w') as f:
            f.write('done')
    return filename[:-1]


def rouge_validation(cfg, task, trainer, val_file='./xsum/val', epoch=1):
    if distributed_utils.is_master(cfg.distributed_training):
        logger.info('start rouge val by {} ...'.format(val_file))
        val_dir = os.path.join(cfg.checkpoint.save_dir, 'val_dir')
        prev_files = glob(os.path.join(val_dir, '*'))
        for j in prev_files:
            os.remove(j)
            print('removed prev files {}..'.format(j))
    # val_file='./xsum/val'
    bart = copy.deepcopy(trainer.get_model())
    bart = BARTHubInterface(cfg, task, bart)
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'
    bart.cuda()
    bart.eval()
    bart.half()

    if distributed_utils.is_master(cfg.distributed_training):
        logger.info('val hypo gen..')
    val_file_name = summa_gen(cfg, val_file + '.source', bart)

    R2_loss = 10e10
    """
    with torch.no_grad():
        hypotheses_batch = bart.sample(valsrc, beam=beam, lenpen=lenpen,
                                       max_len_b=max_len_b,
                                       min_len=min_len, no_repeat_ngram_size=3)
    """
    if distributed_utils.is_master(cfg.distributed_training):
        with open(val_file_name) as f, open(val_file + '.target') as tgt:
            valtgt = tgt.readlines()
            hypotheses_batch = f.readlines()
        assert (len(hypotheses_batch) == len(valtgt)), "val data mismatch hypo, val {}, {}".format(
            len(hypotheses_batch), len(valtgt))
        R2_scores = []
        R_scores = []
        rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"], stats=['f'])
        alpha = [0.05, 0.475, 0.475]
        for i in range(len(hypotheses_batch)):
            scores = rouge.get_scores(hypotheses_batch[i], valtgt[i])
            alpha_score = alpha[0] * scores[0]['rouge-1']['f'] + \
                          alpha[1] * scores[0]['rouge-2']['f'] + \
                          alpha[2] * scores[0]['rouge-l']['f']
            # R2_scores.append(scores[0]['rouge-1']['f'])
            R2_scores.append(alpha_score)
            R_scores.append(scores)
        R2_loss = 10. - np.array(R2_scores).mean() * 10
        print('val examples ..')
        idxs = random.sample(list(range(len(hypotheses_batch))), 3)
        for i in idxs:
            print('*' * 70)
            print(hypotheses_batch[i])
            print('-' * 70)
            print(valtgt[i])
            print(R_scores[i][0])
        # logger.info('current Rouge val loss : {}'.format(R2_loss))
    del bart
    return R2_loss


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
        rouge_val=False,
        orig_state=None
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
            cfg.optimization.stop_time_hours > 0
            and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )
    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                          or should_stop
                          or (
                                  cfg.dataset.validate_interval_updates > 0
                                  and num_updates > 0
                                  and num_updates % cfg.dataset.validate_interval_updates == 0
                          )
                  ) and not cfg.dataset.disable_validation

    # Validate
    valid_losses = [None]

    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets, orig_state=orig_state)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
        model=None,
        orig_state=None
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)

        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample, prev_state=orig_state)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
        cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
