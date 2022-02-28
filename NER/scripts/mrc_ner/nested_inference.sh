#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/leesm/Project/semeval2022_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=semeval
DATA_DIR=/home/leesm/Project/semeval2022_ner/data/semeval
BERT_DIR=/home/leesm/Project/semeval2022_ner
MAX_LEN=512
MODEL_CKPT=/home/leesm/Project/semeval2022_ner/output/bioprocess_sci_uncased/epoch=304.ckpt
HPARAMS_FILE=/home/leesm/Project/semeval2022_ner/output/bioprocess_sci_uncased/lightning_logs/version_1/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}