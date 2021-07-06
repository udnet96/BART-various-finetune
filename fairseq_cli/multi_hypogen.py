import torch
from fairseq.models.bart import BARTModel
import time, os, sys, copy
from sms import mes
from oneToMany import o2m
from pyrouge_test import pyrouge_go
from datetime import datetime
import argparse
import logging
import math
from glob import glob
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

rootlogger = make_logger(__file__)


def check(cfg: DictConfig) -> range:
    save_dir = cfg.checkpoint.save_dir
    if distributed_utils.is_master(cfg.distributed_training):
        devices = [i for i in range(1, cfg.distributed_training.distributed_world_size)]
        conds = []
        for i in devices:
            cond = os.path.join(save_dir, 'valgendone{}'.format(cfg.distributed_training.distributed_rank))
            conds.append(cond)
        while True:
            done = 0
            for cond in conds:
                if os.path.exists(cond):
                    done += 1
            if done >= cfg.distributed_training.distributed_world_size:
                break
        for cond in conds:
            if os.path.exists(cond):
                os.remove(cond)
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


def range_divide(cfg: DictConfig, div_len=0) -> range:
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


def summa_gen2(cfg: DictConfig, target_src='./xsum/test.source',
               ckptfile='./checkpoints/bart.large.xsum/model.pt') -> str:
    src = target_src
    ckptfile = cfg.checkpoint.restore_file
    print('ckpt file : ', ckptfile)
    # try:
    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)
    save_dir = os.path.join(cfg.checkpoint.save_dir, 'hypo_dir')

    start_time = time.time()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = os.path.join(save_dir, 'hypofor.test')
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
    rnge = range_divide(cfg, len(src_lines))

    bart = BARTModel.from_pretrained(
        './',
        checkpoint_file=ckptfile,
        data_name_or_path='xsum-bin_orig'
    )
    bart.cuda()
    bart.eval()
    bart.half()

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


def summa_gen(cfg: DictConfig) -> None:
    dir_name = './checkpoints/bl_200_100_R1init_drop0.1_lr3.0e-05_wu600_bsz16_ufreq3/bl_200_100_R1init_new/'
    # dir_name = './checkpoints/bl_200_100_byR1stop'
    epochs = range(1, 2)
    # src_file = 'test.source'
    # src_file = 'train_low.source'
    src_file = 'val_30.source'
    mode = 'xsum'
    cal_bertscore, cal_rouge = False, False
    # try:
    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)
    save_dir = cfg.checkpoint.save_dir
    cnt = 0
    for ep in epochs:
        cnt += 1
        start_time = time.time()
        epoch = ep
        best, last = True, False
        if best:
            which_ckpt = '_best'
        elif last:
            which_ckpt = '_last'
        else:
            which_ckpt = epoch
        d_name = os.path.dirname(dir_name).split('/')[-1]
        if not os.path.exists(d_name):
            os.mkdir(d_name)
        filename = os.path.join(save_dir, '{}_{}.hypo'.format(d_name, which_ckpt))

        ckptfile = os.path.join(dir_name, 'checkpoint{}.pt'.format(which_ckpt))
        filename += str(cfg.distributed_training.distributed_rank)

        if mode == 'cnn':
            beam, lenpen, max_len_b, min_len, data = 4, 2.0, 140, 55, 'cnndm'
        elif mode == 'gigaword':
            beam, lenpen, max_len_b, min_len, data = 6, 1.0, 20, 0, 'gigaword'
        else:
            beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'

        src = os.path.join(data, src_file)
        if distributed_utils.is_master(cfg.distributed_training):
            print('[{}]\nfile | [{}] gen by\nckpt | [{}] start.. \nsrc | [{}]'.format(datetime.now(),
                                                                                      filename, ckptfile,
                                                                                      src))
            print('mode | {}, beam | {}, lenpen | {}\nmax_len | {}, min_len | {}'.
                  format(data, beam, lenpen, max_len_b, min_len))
        with open(src, 'r', encoding='utf8') as source:
            src_lines = source.readlines()
        rnge = range_divide(cfg, len(src_lines))

        bart = BARTModel.from_pretrained(
            './',
            checkpoint_file=ckptfile,
            data_name_or_path='xsum-bin_low_200_100_ver2'
        )
        bart.cuda()
        bart.eval()
        bart.half()

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
        # mes('summa gened to {} in {} min.'.format(os.path.basename(filename), str(fintime)))
        del bart, hypotheses_batch, slines
        torch.cuda.empty_cache()
        if distributed_utils.is_master(cfg.distributed_training):
            a = sorted(glob(filename[:-2] + '*'))
            with open(filename[:-1], 'w', encoding='utf8') as fout:
                for j in a:
                    if j != filename[:-1]:
                        with open(j) as fin:
                            contents = fin.read().strip()
                            fout.write(contents.strip() + '\n')
        print('rank {} done.'.format(cfg.distributed_training.distributed_rank))

        # rouge score calcul
        if cal_rouge:
            try:
                print('start o2m')
                print('filename : ', filename)
                rouge_results, wk_time, res_file = pyrouge_go(o2m(filename, onefile_dir='gigaword'))
            except Exception as e:
                mes('error during rouge result: ' + str(e))
                print('error during rouge result: ', e)
    if cnt >= 2:
        mes('all summa done.')


def print_hello(cfg: DictConfig) -> None:
    print('world hello!', cfg.distributed_training.distributed_rank)


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)
    # target_func = main
    target_func = summa_gen2
    #target_func = check
    # target_func = print_hello
    # target_func = range_devide

    backend = cfg.distributed_training.distributed_backend
    #cfg.distributed_training.distributed_init_method = 'tcp://10.1.1.20:23456'
    init_method = cfg.distributed_training.distributed_init_method
    world_size = cfg.distributed_training.distributed_world_size
    rank = cfg.distributed_training.distributed_rank
    print(backend, init_method, world_size, rank)
    num_gpu = cfg.distributed_training.distributed_world_size
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, target_func)
    else:
        distributed_utils.call_main(cfg, target_func)


if __name__ == "__main__":
    cli_main()
"""import torch.distributed as dist

backend='nccl'
world_size=2
rank=0
init_method='tcp://10.1.1.20:23456'
dist.init_process_group(backend, world_size=world_size, init_method=None, rank=rank)"""
