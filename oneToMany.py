import struct, re, os, glob
from pyrouge_test import pyrouge_go
import os, sys, time
from sms import mes

dataset = 'xsum'


def o2m(target_dir, ref_file='test.target', onefile_dir='./xsum', sys_dir='./rouge_results'):
    path2 = os.path.join(sys_dir, target_dir.split('/')[-1])
    if not os.path.exists(path2):
        os.makedirs(path2)
    ref_path = os.path.join(path2, 'ref')
    sys_path = os.path.join(path2, 'sys')

    if not os.path.exists(ref_path):
        os.makedirs(ref_path)
    if not os.path.exists(sys_path):
        os.makedirs(sys_path)

    tgt = os.path.join(onefile_dir, ref_file)
    t = glob.glob(tgt)
    src = os.path.join(onefile_dir, '*' + target_dir + '*')
    s = glob.glob(src + '*')
    if len(s) < 1:
        src = target_dir
        if os.path.exists(src):
            s = [src]
        else:
            s = glob.glob(src + '*')
    if len(t) < 1:
        t = glob.glob(ref_file + '*')
    print('rouge gold = ', t)
    print('rouge hypo = ', s)

    # print('ref, sys building ..')
    with open(t[0], 'r', encoding='utf8') as file:
        with open(s[0], 'r', encoding='utf8') as file2:
            l2 = file2.readlines()
            l = file.readlines()
            for i, d in enumerate(l2):
                with open(os.path.join(sys_path, '%06d_sys.txt') % i, 'w', encoding='utf8') as f:
                    sys_w = l2[i].replace('<q>', '. ')
                    f.write(sys_w)
                with open(os.path.join(ref_path, '%06d_ref.txt') % i, 'w', encoding='utf8') as f:
                    ref_w = l[i].replace('<q>', '. ')
                    f.write(ref_w)
    return path2


if __name__ == '__main__':
    try:
        server = '131 orig : '
        gold_label = 'train_unsup_2000_for_eval.target'
        hypo_file = './hypos/ST_test/train_unsup_2000_blx_gen.target'
        print('start o2m. filename : ', hypo_file)
        rouge_results, wk_time, res_file = pyrouge_go(
            o2m(hypo_file, gold_label, 'hypos/ST_test/', 'hypos/ST_test/rouge_results_train'))
        R_result = '%s : ' % server + str(
            hypo_file) + ' rouge result.\n' + rouge_results + '\nin %.2f min' % (wk_time)
        print(R_result)
        mes(R_result)
    except Exception as e:
        mes('error during rouge result: ' + str(e))
        print('error : ', e)
        sys.exit()
    sys.exit()
