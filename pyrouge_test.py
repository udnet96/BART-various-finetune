import pyrouge
import logging
from sms import mes, mes2
import time
import os
import shutil

sys_dir = './rouge_results'


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    print('rouge_eval start')
    # try:
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_ref.txt'
    r.system_filename_pattern = '(\d+)_sys.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    # except Exception as e:
    # mes('rouge_eval failed : ' + str(e))
    # print(rouge_results)
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write, name, start_time=0):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    short_log_str = ""
    short_log_str_all = ""
    print('rouge_log start')
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        short_log_str_all += "R%s:" % x
        for y in ["recall", "precision", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
            short_log_str_all += " %.4f " % (val)
            if y == "f_score":
                short_log_str += "R%c: %.4f " % (x, val)
        short_log_str_all += "\n"

    print(log_str)
    results_file = os.path.join(dir_to_write, "{}".format(name))
    print("Writing final ROUGE results to %s.." % results_file)
    end = time.time()
    with open(results_file, "w") as f:
        f.write(log_str)
        f.write('working time = {} min'.format((end - start_time) / 60.))

    return log_str, results_file, short_log_str, short_log_str_all


def pyrouge_go(path, res_dir='rouge_results', mes_to=1):
    start_time = time.time()
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    dec_dir = os.path.join(path, 'sys')
    ref_dir = os.path.join(path, 'ref')
    results_dict = rouge_eval(ref_dir, dec_dir)
    # print('results_dict = ', results_dict)
    res_file_name = 'rouge_results.txt'
    rouge_results, res_file, short_rouge_results, short_R_results_all = rouge_log(results_dict, path, res_file_name,
                                                                                  start_time)
    end = time.time()
    # print('remove sys, ref folders ...')
    shutil.rmtree(dec_dir)
    shutil.rmtree(ref_dir)
    wk_time = (end - start_time) / 60.
    R_result = [str(res_file.split(os.path.basename(res_file))[0]), 'rouge result.\n', short_R_results_all,
                '\nin %.2f min' % (wk_time), os.path.basename(__file__)]
    R_result = (' | ').join(R_result)
    if mes_to == 1:
        mes(R_result)
    else:
        mes2(R_result)
    print('rouge Working time : %.2f min' % (wk_time))

    return short_rouge_results, wk_time, res_file


if __name__ == '__main__':
    print('pyrouge go')
    g = os.path.join(sys_dir, 'tb')
    pyrouge_go(g)
