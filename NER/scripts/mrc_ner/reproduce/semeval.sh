#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: ace04.sh

TIME=0910
REPO_PATH=/home/leesm/Project/semeval2022_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/home/leesm/Project/semeval2022_ner/data/semeval
BERT_DIR=/home/leesm/Project/semeval2022_ner

BERT_DROPOUT=0.1
MRC_DROPOUT=0.1
LR=3e-5
SPAN_WEIGHT=0.1
WARMUP=500
MAXLEN=512
MAXNORM=1.0
INTER_HIDDEN=1024

BATCH_SIZE=1
PREC=16
VAL_CKPT=0.5
ACC_GRAD=10
MAX_EPOCH=4000
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1

OUTPUT_DIR=/home/leesm/Project/semeval2022_ner/output/large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--gpus="4" \
--distributed_backend=ddp \
--workers 3 \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--gradient_clip_val ${MAXNORM} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

