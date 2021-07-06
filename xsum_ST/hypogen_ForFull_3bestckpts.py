import torch
from fairseq.models.bart import BARTModel
import time, os, sys
from sms import mes
from oneToMany import o2m
from pyrouge_test import pyrouge_go
from datetime import datetime
from glob import glob
import sys

f_name = os.path.basename(__file__)


def hypogen(ckpt, iter=0, is_test=False, num_to_write=-1, cal_rouge=True, summa_gen=True):
    dir_name_base = ckpt.split('/')[-3]
    bsz = 36  # for 12GB GPU memory
    print('hypogen start..')
    cal_rouge, summa_gen = cal_rouge, summa_gen

    if is_test:
        src_file = './xsum/test.source'
        tgt_dir = './xsum_ST/hyps_test/{}'.format(dir_name_base)
    else:
        src_file = './xsum/train_unsup.source'
        tgt_dir = './xsum_ST/hyps/{}'.format(dir_name_base)
    print('src file :', src_file)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    if num_to_write < 0:
        with open(src_file) as f:
            n_to_write = len(f.readlines())
    else:
        n_to_write = num_to_write
    beam, lenpen, max_len_b, min_len, data = 6, 1.0, 60, 0, 'xsum'
    # try:
    start_time = time.time()
    #filename = os.path.join(tgt_dir,
    #                        'unsup_%d_iter%d.hypo' % (num_to_write, iter))
    filename = os.path.join(tgt_dir,
                            '{}.hypo'.format(os.path.basename(ckpt)))
    print('ckpt : {}\nwrite file : {}'.format(ckpt, filename))
    print(datetime.now())
    print('mode : {}, beam : {}, lenpen : {}\nmax_len : {}, min_len : {}'.format(data, beam, lenpen, max_len_b,
                                                                                 min_len))
    # model gen summary write
    if summa_gen:
        bart = BARTModel.from_pretrained(
            './', checkpoint_file=ckpt, data_name_or_path='xsum-bin_low_200_100_ver2'
        )
        bart.cuda()
        bart.eval()
        bart.half()
        count = 1
        print('summa gen start ..')
        with open(src_file, 'r', encoding='utf8') as source, open(filename, 'w',
                                                                  encoding='utf8') as fout:
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
                if count >= n_to_write:
                    break
            if slines != []:
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b,
                                               min_len=min_len, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()

        del bart, hypotheses_batch, slines
        torch.cuda.empty_cache()

    # rouge score calcul
    if cal_rouge:
        try:
            gold_label = './xsum/{}.target'.format(os.path.basename(src_file).split('.')[0])
            if num_to_write > 0:
                true_gold_file = './xsum_ST/{}_{}.target'.format(os.path.basename(src_file).split('.')[0],
                                                                 num_to_write)
                if not os.path.exists(true_gold_file):
                    with open(true_gold_file, 'w') as fout, open(gold_label) as fin:
                        gold_lines = fin.readlines()
                        for line in gold_lines[:n_to_write]:
                            fout.write(line)
            else:
                true_gold_file = gold_label
            hypo_file = filename
            print('start o2m. filename : ', hypo_file)
            if is_test:
                sys_dir = 'xsum_ST/rouge_results_test/{}'.format(dir_name_base)
            else:
                sys_dir = 'xsum_ST/rouge_results_train/{}'.format(dir_name_base)
            if not os.path.exists(sys_dir):
                os.makedirs(sys_dir)
            rouge_results, wk_time, res_file = pyrouge_go(
                o2m(hypo_file, true_gold_file, onefile_dir='xsum_ST/',
                    sys_dir=sys_dir))
            R_result = str(
                hypo_file) + ' rouge result.\n' + rouge_results + '\nin %.2f min' % (wk_time)
            print(R_result)
            # mes(R_result)
        except Exception as e:
            mes('error during rouge result: ' + str(e))
            print('error during rouge result: ', e)
    end_time = time.time()
    fintime = (end_time - start_time) / 60.0
    print('finished %s in ' % filename.split('/')[-1], fintime, 'min')
    mes('{} gen done in {:0.2f}min. | {}'.format(filename, fintime, f_name))
    return filename


hypo_cp_name = './xsum_ST/data/pseudo_data_forbin.target'
if __name__ == '__main__':
    # for generation
    if len(sys.argv) > 4:
        hypofile = hypogen(sys.argv[1], int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]))
        print('hypo written at {}'.format(hypofile))
        if len(sys.argv) > 5:
            hypo_cp_name = sys.argv[5]
        os.system('cp {} {}'.format(hypofile, hypo_cp_name))
    elif len(sys.argv) >= 3:
        hypofile = hypogen(sys.argv[1], int(sys.argv[2]), bool(sys.argv[3]))
