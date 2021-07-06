#!/usr/bin/env bash
bbin=$1
target_func=$2
echo $0 $1
DEVICE=0,2,3,4,7
TOTAL_NUM_UPDATES=30000
TRAIN_STEPS=10000
MAX_TOKENS=1024
SEED=${RANDOM}
LR=3.0e-05
FT_DROPOUT=0.1
FT_BSZ=16
FT_UPDATE_FRQ=3
FT_WARMUP_UPDATES=400
FT_MAX_EPOCH=100
FT_PATIENCE=20
#BART_PATH=./checkpoints/bart.large.cnn/model.pt
BART_PATH=./xsum_ST/ckpts_init/bl_200_100_Ralphastop_cosloss_drop0.1_lr3.0e-05_wu400_bsz16_ufreq3/bl_200_100_Ralphastop_cosloss/checkpoint_best.pt
#BART_PATH=./checkpoints/blx_drop0.1_lr3.0e-05_wu500_bsz512_ufreq4/full_training2/checkpoint_best.pt
SAVE_DIR1=./xsum_ST/ckpts_init/bl_200_100_Ralphastop_cosloss_drop0.1_lr3.0e-05_wu400_bsz16_ufreq3/bl_200_100_Ralphastop_cosloss/
TRAIN_FILE=fairseq_cli/multi_hypogen.py
TRAIN_CRITERION=label_smoothed_cross_entropy
#TRAIN_CRITERION=semantic_similarity_loss
if [ ! -d $SAVE_DIR1 ]; then
  mkdir $SAVE_DIR1
fi
SAVE_DIR=${SAVE_DIR1}
if [ ! -d $SAVE_DIR ]; then
  mkdir $SAVE_DIR
fi

CUDA_VISIBLE_DEVICES=${DEVICE} python ${TRAIN_FILE} ${bbin} \
  --memory-efficient-fp16 \
  --max-epoch ${FT_MAX_EPOCH} \
  --max-update ${TRAIN_STEPS} \
  --save-dir ${SAVE_DIR} \
  --batch-size ${FT_BSZ} --seed ${SEED} \
  --update-freq ${FT_UPDATE_FRQ} \
  --keep-best-checkpoints 1 \
  --no-last-checkpoints \
  --keep-last-epochs 1 \
  --patience ${FT_PATIENCE} \
  --restore-file ${BART_PATH} \
  --max-tokens ${MAX_TOKENS} \
  --task translation \
  --source-lang source --target-lang target \
  --truncate-source \
  --layernorm-embedding \
  --share-all-embeddings \
  --share-decoder-input-output-embed \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --arch bart_large \
  --criterion ${TRAIN_CRITERION} \
  --label-smoothing 0.1 \
  --dropout ${FT_DROPOUT} --attention-dropout ${FT_DROPOUT} \
  --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
  --clip-norm 0.1 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES \
  --warmup-updates ${FT_WARMUP_UPDATES} \
  --fp16 \
  --skip-invalid-size-inputs-valid-test \
  --find-unused-parameters --bpe gpt2;
wait $!
CUDA_VISIBLE_DEVICES=${DEVICE} python R_test.py ${SAVE_DIR} True

:<<END
ITERATION=5
DEVICE=2
ckpt=./xsum_ST/ckpts/bl_ptft_xent_srcnoise_chagned_2000/iter${ITERATION}_FT_drop0.1_nsteps30000/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_indep.py ${ckpt} ${ITERATION} True
END