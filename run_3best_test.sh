#!/usr/bin/env bash
DEVICE=4
LIST="
4.12
4.16
"
# example usage
for F in $LIST;
do
SAVE_DIR=./checkpoints/xsum2__seed200__fromblx_r3f_cossim0.3_adam_xentstop_dev3/fromblx_xentstop_r3f_cossim0.3/checkpoint.best_loss_${F}.pt
CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_ForFull_3bestckpts.py ${SAVE_DIR} 0 True;
done
python sms.py ${DEVICE}
