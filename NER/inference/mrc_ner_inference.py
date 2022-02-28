#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_inference.py

import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
set_random_seed(0)
from train.mrc_ner_trainer import BertLabeling
from datasets.mrc_ner_dataset import MRCNERDataset_infer
from metrics.functional.query_span_f1 import extract_nested_spans
from transformers import BertTokenizerFast
import tqdm

def get_dataloader(config, data_prefix="test"):
    data_path = os.path.join(config.data_dir, f"mrc-ner.{data_prefix}")

    tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[unused10]']})
    dataset = MRCNERDataset_infer(json_path=data_path,
                            tokenizer=tokenizer,
                            max_length=config.max_length,
                            is_chinese=config.is_chinese,
                            pad_to_maxlen=False,
                            mode=data_prefix,
                            data_dir=config.data_dir
                            )

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    return dataloader, tokenizer

def get_query_index_to_label_cate(dataset_sign):
    if dataset_sign == "semeval":
        return {1: "SYM", 2: "PRI", 3: "ORD"}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--flat_ner", action="store_true",)
    parser.add_argument("--dataset_sign", type=str, choices=["semeval"], default="semeval")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    trained_mrc_ner_model = BertLabeling.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=None,
        batch_size=1,
        max_length=args.max_length,
        workers=1)

    data_loader, data_tokenizer = get_dataloader(args,)
    
    test_data_dic = json.load(open(os.path.join(args.data_dir, f"mrc-ner.test"), encoding="utf-8"))
    
    for id in test_data_dic:
        test_data_dic[id]['entity'] = {}
        test_data_dic[id]['relation'] = {}

    # query2label_dict = get_query_index_to_label_cate(args.dataset_sign)

    for batch in tqdm.tqdm(data_loader):
        tokens, token_type_ids, start_label_mask, end_label_mask, doc_id, label, start_at_doc, all_tok_to_char_list = batch

        label = label[0].item()
        
        if label == 5888: # 'symbol - token_id
            label = 'SYMBOL'
        elif label == 3756: # 'description' token_id
            label = 'PRIMARY'         

        attention_mask = (tokens != 0).long()

        start_logits, end_logits, span_logits = trained_mrc_ner_model.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0

        subtokens_idx_lst = tokens.numpy().tolist()[0]
        subtokens_lst = data_tokenizer.convert_ids_to_tokens(subtokens_idx_lst)# [idx2tokens[item] for item in subtokens_idx_lst]
        # label_cate = query2label_dict[label_idx.item()]
        readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True)

        match_preds = span_preds
        entities_info = extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, pseudo_tag="TAG")#label_cate)

        entity_lst = []
        entity_char_lst = []
        

        if len(entities_info) != 0:
            for entity_info in entities_info:
                start, end = entity_info[0], entity_info[1]
                entity_string = data_tokenizer.convert_tokens_to_string(subtokens_lst[start:end+1])

                entity_lst.append((start, end+1, entity_string, entity_info[2]))

                context = test_data_dic[doc_id[0]]['text']
                
                start_char_idx = all_tok_to_char_list[start - 3 + start_at_doc.item()][0].item() 
                end_char_idx = all_tok_to_char_list[end - 3  + start_at_doc.item()][1].item()
                entity_char_lst.append((start_char_idx, end_char_idx, context[start_char_idx:end_char_idx]))
                
                entity_idx = len(test_data_dic[doc_id[0]]['entity']) + 1
                
                flag = False
                for entity_name in test_data_dic[doc_id[0]]['entity']:
                    if test_data_dic[doc_id[0]]['entity'][entity_name]['start']==start_char_idx and test_data_dic[doc_id[0]]['entity'][entity_name]['end']==end_char_idx and test_data_dic[doc_id[0]]['entity'][entity_name]['label']==label:
                        flag = True
                        break
                if flag:
                    continue
                                
                test_data_dic[doc_id[0]]['entity'][f'T{entity_idx}'] = \
                    {"eid": f'T{entity_idx}', "label": label, "start": start_char_idx, "end": end_char_idx, \
                        "text": context[start_char_idx:end_char_idx]}

        # print("*="*10)
        
        # print(f"Given input: {readable_input_str}")
        # print(f"Model predict: {entity_lst}")
        # print(f"Model predict [CHAR]: {entity_char_lst}")
    
    with open('./mrc-ner.test.ner_result', 'w', encoding='utf-8') as make_file:
        json.dump(test_data_dic, make_file, indent="\t")
if __name__ == "__main__":
    main()