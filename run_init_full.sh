#!/bin/bash

FILE_NAME=$0
bbin=$1

if [ $# -ne 1 ] ; then
    echo "bash ${0} bin"
    exit 0
fi

DEVICE=3
echo "DEVICE : ${DEVICE}"
#RESIDUAL_CHANGE=2step_reduced0.75_LNatonce

TIMESTEP=1
while [ $TIMESTEP -le 1 ]
do
echo "timestep : ${TIMESTEP}"
#SEED=${RANDOM}
SEED=$(echo "${TIMESTEP}*100" | bc)
echo "SEED : ${SEED}"

TOTAL_NUM_UPDATES=30000
TRAIN_STEPS=10000000
VERSION=${TIMESTEP}

model_from=blx
#model_from=31.3
#model_from=33.3
#model_from=bl
#model_from=43.62
#model_from=43.81

#BART_PATH=./checkpoints/bart.large.cnn/model.pt
#BART_PATH=./checkpoints/bart.large/model.pt
#BART_PATH=./checkpoints/blx_exf_norm_bsz90_43.62/checkpoint88.pt
BART_PATH=./checkpoints/bart.large.xsum/model.pt
#BART_PATH=./checkpoints/blc_norm_xentstop_drop0.1_lr3.0e-05_wu500_bsz1_ufreq2/blc_xentstop_norm/checkpoint_best.pt
#BART_PATH=./checkpoints/bl_200_100_byR1stop/checkpoint.best_loss_6.50.pt
#BART_PATH=./checkpoints_low/bl_200_100_Ralphastop_r3f0.1_uniformnoise_drop0.1_lr3.0e-05_wu400_bsz16_ufreq3/bl_200_100_Ralphastop_r3f_33.3/checkpoint_best.pt
#BART_PATH=./checkpoints/blc_norm_xentstop_drop0.1_lr3.0e-05_wu500_bsz1_ufreq2/blc_xentstop_norm_43.81/checkpoint_best.pt

#MAX_TOKENS=2048
MAX_TOKENS=1024
FT_WARMUP_UPDATES=500
#LR=5e-04
#WEIGHT_DECAY=1.0e-04
#OPTIM=adabelief
#ADAM_EPS=1e-16

WEIGHT_DECAY=0.01
LR=3.0e-05
OPTIM=adam
ADAM_EPS=1e-8

FT_DROPOUT=0.1

FT_BSZ=128
FT_UPDATE_FRQ=32
FT_MAX_EPOCH=50
FT_PATIENCE=3

NOISE_TYPE=uniform
R3F_LAMBDA=0.1

STOP_CRITERIA=xentstop
#STOP_CRITERIA=Rstop
TRAIN_FILE=fairseq_cli/train.py
#TRAIN_FILE=fairseq_cli/train_CS.py
#TRAIN_FILE=fairseq_cli/train2.py
#TRAIN_FILE=fairseq_cli/train_midval.py
#TRAIN_FILE=fairseq_cli/train_midval_Rstop.py
#TRAIN_FILE=fairseq_cli/train_rouge_val_low.py
#TRAIN_FILE=fairseq_cli/train_R3F_CS_Rstoplow_midval.py
#TRAIN_FILE=fairseq_cli/train_rougeRL_Rval_midval.py
#TRAIN_FILE=fairseq_cli/train2.py
#TRAIN_FILE=fairseq_cli/train_rouge_val.py
#TRAIN_FILE=fairseq_cli/train_Rval_cnndmlow.py
#TRAIN_FILE=fairseq_cli/train_R3F_cossim_Rstop.py
#TRAIN_FILE=fairseq_cli/train_R3F_cossim_Rstoplow_Sbert.py
#TRAIN_FILE=fairseq_cli/train_Ralpha_val.py

#CRITERIONN=r3f
#CRITERIONN=r3f_wolog
#CRITERIONN=r3f_cossim0.3
#CRITERIONN=cossim0.3
CRITERIONN=norm

TRAIN_CRITERION=label_smoothed_cross_entropy
#TRAIN_CRITERION=label_smoothed_cross_entropy_r3f
#TRAIN_CRITERION=label_smoothed_cross_entropy_r3f_cossim
#TRAIN_CRITERION=label_smoothed_cross_entropy_cossim2
#TRAIN_CRITERION=semantic_similarity_loss

#ATTN_CHANGE=valrate
#RATE=0.25
FT_PATIENCE=3
#FT_PATIENCE=$(echo 5/${RATE} | bc)

#SAVE_DIR1=./checkpoints_low/xsumlow${VERSION}_${RESIDUAL_CHANGE}_seed${SEED}_${ATTN_CHANGE}${RATE}_from${model_from}_${CRITERIONN}_${OPTIM}_${STOP_CRITERIA}_drop${FT_DROPOUT}_lr${LR}_wu${FT_WARMUP_UPDATES}_bsz${FT_BSZ}_ufreq${FT_UPDATE_FRQ}
SAVE_DIR1=./checkpoints/xsum${VERSION}_${RESIDUAL_CHANGE}_seed${SEED}_${ATTN_CHANGE}${RATE}_from${model_from}_${CRITERIONN}_${OPTIM}_${STOP_CRITERIA}_dev${DEVICE}

if [ ! -d $SAVE_DIR1 ]; then
  mkdir $SAVE_DIR1
fi
SAVE_DIR=${SAVE_DIR1}/from${model_from}_${STOP_CRITERIA}_${CRITERIONN}
if [ ! -d $SAVE_DIR ]; then
  mkdir $SAVE_DIR
fi
#  --noise-type ${NOISE_TYPE} --r3f-lambda ${R3F_LAMBDA} \
#  --ddp-backend=no_c10d \

CUDA_VISIBLE_DEVICES=${DEVICE} python ${TRAIN_FILE} ${bbin} \
  --memory-efficient-fp16 \
  --fp16 \
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
  --weight-decay ${WEIGHT_DECAY} --optimizer ${OPTIM} --adam-betas "(0.9, 0.999)" --adam-eps ${ADAM_EPS} \
  --clip-norm 0.1 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES \
  --warmup-updates ${FT_WARMUP_UPDATES} \
  --skip-invalid-size-inputs-valid-test \
  --find-unused-parameters --bpe gpt2;
wait $!

#SAMPLE_SIZE=7330
#CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_Forunsup.py ${SAVE_DIR}/checkpoint_best.pt 0 '' ${SAMPLE_SIZE}
#CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_fortest_cnndm.py ${SAVE_DIR}/checkpoint_best.pt 0 True
#CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_Forunsup.py ${SAVE_DIR}/checkpoint_best.pt 0 True
CUDA_VISIBLE_DEVICES=${DEVICE} python ./xsum_ST/hypogen_ForFull_3bestckpts.py ${SAVE_DIR}/checkpoint_best.pt 0 True;
((TIMESTEP++))
done

python sms.py ${DEVICE}
