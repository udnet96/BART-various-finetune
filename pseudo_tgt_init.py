import os, sys, glob

if __name__=='__main__':
    args = sys.argv[1:]
    init_pseudo = args[0]
    num_to_write = int(args[1])
    init_from = args[2]
    if num_to_write > 0:
        true_file = init_pseudo + '.target'
        gold_label = init_from
        with open(true_file, 'w') as fout, open(gold_label) as fin:
            gold_lines = fin.readlines()
            for line in gold_lines[:num_to_write]:
                fout.write(line)
        print('\n| {} init by {} ! | \n'.format(true_file, gold_label))
