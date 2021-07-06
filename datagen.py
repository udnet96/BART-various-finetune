import os, sys, glob

if __name__=='__main__':
    args = sys.argv[1:]
    src_file = args[0]
    num_to_write = int(args[1])
    gold_label = './xsum/{}.source'.format(os.path.basename(src_file).split('.')[0])
    if num_to_write > 0:
        true_file = './xsum_ST/data/{}_{}.source'.format(os.path.basename(src_file).split('.')[0],
                                                              num_to_write)
        if not os.path.exists(true_file):
            with open(true_file, 'w') as fout, open(gold_label) as fin:
                gold_lines = fin.readlines()
                for line in gold_lines[:num_to_write]:
                    fout.write(line)
        print('\n| {} gened ! |\n'.format(true_file))
