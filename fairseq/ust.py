import torch
import numpy as np
import time, os, sys, random
from glob import glob
from fairseq.models.bart import BARTModel
from torch import nn
import logging

logger = logging.getLogger("fairseq.ust")


# all sents in unsup data pool
def mc_dropout_eval(ckpt_file, sents, epoch=1, T=10, maxlen=120, batch_size=24):
    start_time = time.time()
    logger.info("Yielding predictions looping over by {}...".format(ckpt_file))
    upper_dir_idx = ckpt_file.rfind('/')
    save_dir = ckpt_file[:upper_dir_idx]
    y_T_save = os.path.join(save_dir, 'y_T_list_ep{}'.format(epoch))
    logger.info('y_T_ save : {}'.format(y_T_save))
    if os.path.exists(str(y_T_save) + '.npy'):
        logger.info('y_T loaded from : {}'.format(y_T_save))
        y_T = np.load(str(y_T_save) + '.npy', allow_pickle=True)
    else:
        model = BARTModel.from_pretrained(
            './', checkpoint_file=ckpt_file,
            data_name_or_path='xsum-bin_orig'
        )
        orig_state_dict = []
        for key in model.state_dict():
            orig_state_dict.append(model.state_dict()[key].clone())
        state_dict = model.state_dict().copy()
        drop = nn.Dropout(p=0.005)

        beam, lenpen, max_len_b, min_len, data = 6, 1.0, maxlen, 0, 'xsum'
        model.cuda()
        model.eval()
        model.half()

        y_T = []
        for T_i in range(T):
            time_per_oneT = time.time()
            logger.info('ep {}, T = {}, pseudo hypo gen..'.format(epoch, T_i))
            # change params to dropout-ed
            with torch.no_grad():
                for i, key in enumerate(state_dict):
                    state_dict[key] = drop(orig_state_dict[i])
            model.load_state_dict(state_dict)

            y_pred = []
            # hypo gen by dropout-ed model
            for i in range(0, len(sents), batch_size):
                if i + batch_size <= len(sents):
                    end = i + batch_size
                else:
                    end = len(sents)
                src_tokens = []
                for j in range(i, end):
                    src_tokens.append(model.encode(sents[j]))
                hypo = model.generate(src_tokens, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                                      min_len=min_len, no_repeat_ngram_size=3)
                for k in hypo:
                    score, hypo_tokens = k[0]['score'].item(), list(np.array(k[0]['tokens'].cpu()))
                    y_pred.append((score, hypo_tokens))
            y_T.append(y_pred)
            # restore params to original
            with torch.no_grad():
                for i, key in enumerate(state_dict):
                    state_dict[key] = orig_state_dict[i]
            model.load_state_dict(state_dict)
            end_time = time.time()
            logger.info('ep {}, T = {}, done in {} min..'.format(epoch, T_i, (end_time - time_per_oneT) / 60.))
        y_T = np.array(y_T)  # y_T = (T, len(sents), 2) array
        np.save(y_T_save, y_T)  # save y_T

    # compute mean
    y_mean = np.mean(y_T[:, :, 0], axis=0)
    assert y_mean.shape == (len(sents),)

    # compute majority prediction
    y_pick = y_T[:, :, 0].argmax(axis=0)  # T개 중 argmax idx가 가장 많은 idx로 label
    assert y_pick.shape == (len(sents),)

    # compute variance
    y_var = np.var(y_T[:, :, 0], axis=0)
    assert y_var.shape == (len(sents),)

    end_time = time.time()
    logger.info('ep {}, mc dropout eval done by {} min..'.format(epoch, (end_time - start_time) / 60.))
    return y_mean, y_var, y_pick, y_T


def pseudo_hypo_gen(src, dir, cfg, num_to_write=-1, epoch=1, maxlen=120):
    if num_to_write < 0:
        num_to_write = 1000000
    src_file, dir_name = src, dir
    start_time = time.time()
    logger.info('ckpt target dir : {}'.format(dir_name))
    ck = sorted(glob(os.path.join(dir_name, 'checkpoint' + '*')))
    if len(ck) < 1:
        ck = sorted(glob(os.path.join(dir_name, '*' + '.pt')))[0]
    else:
        ck = ck[0]
    tgt_name = src_file.split('/')[-1].split('.')[0]
    if len(dir_name.split('/')[-1]) <= 2:
        hyp_savedir = './hypos/ST_test/hyps/{}'.format(dir_name.split('/')[-2])
    else:
        hyp_savedir = './hypos/ST_test/hyps/{}'.format(dir_name.split('/')[-1])
    if not os.path.exists(hyp_savedir):
        os.makedirs(hyp_savedir)
    filename = os.path.join(hyp_savedir,
                            'ep%d_best_%s_%d.hypo' % (epoch, tgt_name, num_to_write))
    logger.info('ckpt : {}'.format(ck))
    logger.info('hypo tgt file name : {}'.format(filename))

    if os.path.exists(filename):
        logger.info('{} : already exists'.format(filename))
    else:
        logger.info('hypo gen..')
        bart = BARTModel.from_pretrained(
            './', checkpoint_file=ck, data_name_or_path='xsum-bin_orig'
        )
        beam, lenpen, max_len_b, min_len, data = 6, 1.0, maxlen, 0, 'xsum'

        bart.cuda()
        bart.eval()
        bart.half()

        bsz, count = 36, 1
        with open(src_file, 'r', encoding='utf8') as source, open(filename, 'w', encoding='utf8') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in source:
                if count % bsz == 0:
                    with torch.no_grad():
                        hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                                                       min_len=min_len, no_repeat_ngram_size=3)

                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []

                slines.append(sline.strip())
                count += 1
                if count >= num_to_write:
                    break
            if slines != []:
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                                               min_len=min_len, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
        end_time = time.time()
        fintime = (end_time - start_time) / 60.0
        logger.info('hypo gen finished {} in {} min'.format(filename.split('/')[-1], fintime))
        del bart, hypotheses_batch, slines
    return ck, filename


def data_prep(sampled_src, sampled_hypo):
    with open('./hypos/ST_test/train_low.target') as f, open('./hypos/ST_test/train_low.source') as f2:
        sup_tgt = f.readlines()
        sup_src = f2.readlines()
    output_tgtfile = './hypos/ST_test/train_norm_all.target'
    output_srcfile = './hypos/ST_test/train_norm_all.source'
    with open(output_tgtfile, 'w', encoding='utf8') as tgtout, open(output_srcfile, 'w', encoding='utf8') as srcout:
        for line in sup_tgt:
            tgtout.write(line.strip() + '\n')
        for sample in sampled_hypo:
            tgtout.write(str(sample).strip() + '\n')

        for line in sup_src:
            srcout.write(line.strip() + '\n')
        for sample in sampled_src:
            srcout.write(str(sample).strip() + '\n')
    logger.info('sup + unsup data built.')
    return '.' + output_srcfile.split('.')[1]


def data_prep_ust(sampled_src, sampled_hypo):
    with open('./hypos/ST_test/train_low.target') as f, open('./hypos/ST_test/train_low.source') as f2:
        sup_tgt = f.readlines()
        sup_src = f2.readlines()
    output_tgtfile = './hypos/ST_test/train_ust_all.target'
    output_srcfile = './hypos/ST_test/train_ust_all.source'
    with open(output_tgtfile, 'w', encoding='utf8') as tgtout, open(output_srcfile, 'w', encoding='utf8') as srcout:
        for line in sup_tgt:
            tgtout.write(line.strip() + '\n')
        for sample in sampled_hypo:
            tgtout.write(str(sample).strip() + '\n')

        for line in sup_src:
            srcout.write(line.strip() + '\n')
        for sample in sampled_src:
            srcout.write(str(sample).strip() + '\n')
    logger.info('sup + unsup data built.')
    return '.' + output_srcfile.split('.')[1]


def sample_by_ymean_easiness(X, y_mean, y_var, num_samples, y_T, ckpt):
    model = BARTModel.from_pretrained(
        './', checkpoint_file=ckpt, data_name_or_path='xsum-bin_orig')
    logger.info("Sampling by y_mean")
    y_m = y_mean.astype(float)
    softmax = torch.nn.Softmax(dim=0)
    p_norm = list(np.array(softmax(torch.tensor(y_m))))
    indices = np.random.choice(len(X), num_samples, p=p_norm, replace=False)

    best_hypo_idx = y_T[:, :, 0].argmax(axis=0)
    best_hypo = []
    for i, n in enumerate(y_T.transpose((1, 0, -1))):
        toks = n[best_hypo_idx[i]][1]
        best_hypo.append(model.decode(torch.tensor(toks, dtype=torch.long)))
    best_hypo = np.array(best_hypo)
    X = np.array(X)

    X_s = X[indices]
    y_s = best_hypo[indices]
    w_s = y_var[indices]
    return X_s, y_s, w_s


def sample_by_bald_difficulty(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):
    logger.info("Sampling by easy BALD acquisition function")
    p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
    p_norm = p_norm / np.sum(p_norm)
    indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
    X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices],
           "attention_mask": X["attention_mask"][indices]}
    y_s = y[indices]
    w_s = y_var[indices][:, 0]
    return X_s, y_s, w_s


if __name__ == '__main__':
    model = BARTModel.from_pretrained(
        './checkpoints', checkpoint_file='bart.large.xsum/model.pt',
        data_name_or_path='xsum-bin_orig'
    )
    unsup_source = open('./xsum/train_unsup.source')
    unsup_src = unsup_source.readlines()
    # after mc dropout eval..
    T = 10
    sample_scheme = 'easy_bald_class'
    unsup_size = 2000
    for epoch in range(25):
        if 'uni' in sample_scheme:
            y_mean, y_var, y_T = None, None, None
        elif 'bald' in sample_scheme:
            y_mean, y_var, y_pred, y_T = mc_dropout_eval(model, unsup_src, T=T)
        else:
            print("Error in specifying sample_scheme: One of the 'uni' or 'bald' schemes need to be specified")
            sys.exit(1)

        # sample from unlabeled set
        if 'conf' in sample_scheme:
            conf = True
        else:
            conf = False

        if 'bald' in sample_scheme and 'eas' in sample_scheme:
            f_ = sample_by_ymean_easiness

        if 'bald' in sample_scheme and 'dif' in sample_scheme:
            f_ = sample_by_bald_difficulty

        if 'uni' in sample_scheme:
            print("Sampling uniformly")
            if unsup_size < len(unsup_src):
                indices = np.random.choice(len(unsup_src), unsup_size, replace=False)
                X_batch, y_batch = np.array(unsup_src)[indices], y_pred[indices]
            else:
                X_batch, y_batch = unsup_src, y_pred
            X_conf = np.ones(len(y_batch))
        else:
            # sample_by_ymean_easiness(X, y_mean, y_var, num_samples, y_T, model):
            X_batch, y_batch, X_conf = f_(unsup_src, y_mean, y_var, unsup_size, y_T, model)
        """
        alpha = 0.1
        if not conf:
            print("Not using confidence learning.")
            X_conf = np.ones(len(X_batch['input_ids']))
            print("Weights ".format(X_conf[:10]))
        else:
            print("Using confidence learning ".format(X_conf[:10]))
            X_conf = -np.log(X_conf + 1e-10) * alpha
            print("Weights ".format(X_conf[:10]))
        """

    logger.info("Test accuracy based on best validation loss {}".format(best_test_acc))
    logger.info("Best test accuracy across all self-training iterations {}".format(max_test_acc))
