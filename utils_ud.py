import logging, os, sys, copy
from rouge import Rouge
from sms import mes, make_logger
from fairseq.models.bart.hub_interface import BARTHubInterface
import torch
import numpy as np
from time import time

logger = make_logger('fairseq.utils_ud')


def rouge_validation(cfg, task, trainer, val_file='./xsum/val', epoch=1):
    logger.info('start rouge val by {} ...'.format(val_file))
    bart = copy.deepcopy(trainer.get_model())
    # state_dict = trainer.get_model().state_dict()
    """ 
    ckpt_file = 'checkpoints/test/checkpoint{}.pt'.format(epoch)
    if not os.path.exists(ckpt_file):
        a = sorted(glob('./checkpoints/test/*'))
        if len(a) > 1:
            ckpt_file = a[1]
        else:
            ckpt_file = a[0]
    logger.info('checkpoint : {}'.format(ckpt_file))
    bart = BARTModel.from_pretrained(
        './',
        checkpoint_file=ckpt_file,
        data_name_or_path='giga-bin_low'
    )
    """
    """
    cnt = 0
    for key in bart.state_dict():
        logger.info(
            'key : {}, {}\nvalue : {}'.format(key, bart.state_dict()[key].shape, bart.state_dict()[key]))
        cnt += 1
        if cnt > 2:
            break
    print('-' * 40)
    """
    bart = BARTHubInterface(cfg, task, bart)
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'
    bart.cuda()
    bart.eval()
    bart.half()

    with open(val_file + '.source') as f, open(val_file + '.target') as tgt:
        valsrc = f.readlines()
        valtgt = tgt.readlines()

    with torch.no_grad():
        logger.info('val hypo gen..')
        hypotheses_batch = bart.sample(valsrc, beam=beam, lenpen=lenpen,
                                       max_len_b=max_len_b,
                                       min_len=min_len, no_repeat_ngram_size=3)
    R2_scores = []
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"], stats=['f'])
    print('val examples ..')
    for i in range(len(hypotheses_batch[:2])):
        scores = rouge.get_scores(hypotheses_batch[i], valtgt[i])
        R2_scores.append(scores[0]['rouge-1']['f'])
        print(hypotheses_batch[i])
        print(valtgt[i])
        print(scores[0])
        print('-' * 30)
    R2_loss = 10. - np.array(R2_scores).mean() * 10
    logger.info('current Rouge val loss : {}'.format(R2_loss))
    del bart
    return R2_loss


def rouge_reward(cfg, task, model, sample):
    st_time = time()
    bart = copy.deepcopy(model)
    bart = BARTHubInterface(cfg, task, bart)
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 10, 'xsum'
    bart.cuda()
    bart.eval()
    bart.half()
    src_tokens = sample['net_input']['src_tokens']
    src_batch = bart.decode(src_tokens)
    print('decoded! : {}\n{}'.format(src_tokens.size, src_batch[0]))

    with torch.no_grad():
        logger.info('val hypo gen..')
        hypotheses_batch = bart.sample(src_batch, beam=beam, lenpen=lenpen,
                                       max_len_b=max_len_b,
                                       min_len=min_len, no_repeat_ngram_size=3)
    ed_time = (st_time - time())/60.
    print('model gen : {}\n{}'.format(hypotheses_batch.size, hypotheses_batch[0]))
    print('time : {}min'.format(ed_time))
    dd

