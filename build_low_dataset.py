import nltk

dataset = 'cnndm'
# low resource sample selection method check
t_src = './cnndm/test.source'
t_tgt = './cnndm/test.target'
tr_src = './cnndm/train.source'
tr_tgt = './cnndm/train.target'
v_src = './cnndm/val.source'
v_tgt = './cnndm/val.target'
tr_src_selected = './cnndm/train_selected_ver3.source'
tr_tgt_selected = './cnndm/train_selected_ver3.target'
joint_src = './cnndm/train_unsup_sup_joint.source'
joint_tgt = './cnndm/train_unsup_sup_joint.target'
vl_src = './cnndm/val_low.source'
vl_tgt = './cnndm/val_low.target'

summ = 0
summ2 = 0
src_max, src_min = 0, 10000
tgt_max, tgt_min = 0, 10000
Full = True
val_check = True
if Full:
    with open(tr_src) as f, open(tr_tgt) as f2, open(v_src) as f3, open(v_tgt) as f4, open(t_src) as f5, open(
            t_tgt) as f6:
        lines = f.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        lines4 = f4.readlines()
        lines5 = f5.readlines()
        lines6 = f6.readlines()
        l1 = lines + lines3 + lines5
        l2 = lines2 + lines4 + lines6
elif val_check:
    with open(vl_src) as f3, open(vl_tgt) as f4:
        lines3 = f3.readlines()
        lines4 = f4.readlines()
        l1 = lines3
        l2 = lines4
else:
    with open(joint_src) as f, open(joint_tgt) as f2, open(vl_src) as f3, open(vl_tgt) as f4:
        lines = f.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        lines4 = f4.readlines()
        l1 = lines + lines3
        l2 = lines2 + lines4
    """
    for line in lines:
        src_len = len(nltk.word_tokenize(line))
        summ += len(nltk.word_tokenize(line))
        if src_len < src_min:
            src_min = src_len
            print('min : ', src_min)
        if src_len > src_max:
            src_max = src_len
            print('max : ', src_max)
    for line in lines2:
        tgt_len = len(nltk.word_tokenize(line))
        summ2 += tgt_len
        if tgt_len < tgt_min:
            tgt_min = tgt_len
            print('tgt min : ', tgt_min)
        if tgt_len > tgt_max:
            tgt_max = tgt_len
            print('tgt max : ', tgt_max)
    avg = summ // len(lines)
    avg2 = summ2 // len(lines2)
    """
srclen_list = []
tgtlen_list = []
for line in l1:
    src_len = len(nltk.word_tokenize(line))
    srclen_list.append(src_len)
    summ += src_len
    if src_len < src_min:
        src_min = src_len
        print('min : ', src_min)
    if src_len > src_max:
        src_max = src_len
        print('max : ', src_max)
for line in l2:
    tgt_len = len(nltk.word_tokenize(line))
    tgtlen_list.append(tgt_len)
    summ2 += tgt_len
    if tgt_len < tgt_min:
        tgt_min = tgt_len
        print('tgt min : ', tgt_min)
    if tgt_len > tgt_max:
        tgt_max = tgt_len
        print('tgt max : ', tgt_max)
srclen_avg = summ // len(l1)
tgtlen_avg = summ2 // len(l2)
print('src avg, min, max :', srclen_avg, src_min, src_max)
print('tgt avg, min, max :', tgtlen_avg, tgt_min, tgt_max)
srclen_list.sort()
tgtlen_list.sort()
print('src 90, 10 %-ile : ', srclen_list[int(len(srclen_list) * 0.9)], srclen_list[int(len(srclen_list) * 0.1)])
print('tgt 90, 10 %-ile : ', tgtlen_list[int(len(tgtlen_list) * 0.9)], tgtlen_list[int(len(tgtlen_list) * 0.1)])

print('src Median : ', srclen_list[int(len(srclen_list) * 0.5)])
print('tgt Median : ', tgtlen_list[int(len(tgtlen_list) * 0.5)])


# print('summ, avg, len : ', summ, avg, len(lines))
# print('summ, avg, len for tgt : ', summ2, avg2, len(lines2))
selected_idx = []

# build low dataset by avg srclen & tgtlen (error rate +_ 10%)
avg = srclen_avg
avg2 = tgtlen_avg
with open(tr_src) as f, open(tr_tgt) as f2, open(tr_src_selected, 'w', encoding='utf8') as fs, \
        open(tr_tgt_selected, 'w', encoding='utf8') as fs2:
    src_lines = f.readlines()
    tgt_lines = f2.readlines()
    for i, line in enumerate(src_lines):
        tklen = len(nltk.word_tokenize(line))
        tklen2 = len(nltk.word_tokenize(tgt_lines[i]))
        if (avg - (avg * 0.1) < tklen <= avg + (avg * 0.1)) and (
                (avg2 - (avg2 * 0.1) <= tklen2 <= avg2 + (avg2 * 0.1))):
            selected_idx.append(i)
            print('{} '.format(i), end='')
            print(line.strip(), file=fs)
            print(tgt_lines[i].strip(), file=fs2)

print('# selected samples :', len(selected_idx))
