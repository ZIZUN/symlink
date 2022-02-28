import re
from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import AutoTokenizer
from util.dataset.prepro_re import prepro_re
import json

class LoadDataset(Dataset):
    def __init__(self, corpus_path, seq_len, mode='train', model_name='scibert'):
        self.seq_len = seq_len

        self.corpus_path = corpus_path
        
        if model_name=='scibert':
            self.tokenizer = AutoTokenizer.from_pretrained("./pretrained_model/scibert_cased/")
        elif model_name=='scibert_uncased':
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        elif model_name=='deberta_v3_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        elif model_name=='deberta_v2_xxlarge':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")            
        elif model_name=='deberta_v2_xlarge':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")   
               
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[unused10]']})
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad = self.tokenizer.pad_token_id

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<S>', '</S>']})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<P>', '</P>']})

        tokens_map = {'SYMBOL': ['<S>', '</S>'], 'PRIMARY': ['<P>', '</P>']}
        label_map = {'Count': 0, 'Direct': 1, 'Corefer-Symbol': 2, 'Corefer-Description':3, 'Negative_Sample':4}
        
        self.all_data = json.load(open(corpus_path, encoding="utf-8"))
        self.examples = prepro_re(docs=self.all_data, tokenizer=self.tokenizer, maxlen=seq_len, mode=mode)
        
        # == [span_1] [span_2] ==
        # primary -> symbol : direct
        # primary -> symbol : count
        # primary <-> primary : corefer-to
        # symbol <-> symbol : corefer-to
        # else : None

        self.dataset_len = len(self.examples)

        self.dataset = []
        for i in range(self.dataset_len):
            example = self.examples[i]

            ids, span_1, span_2 = self.tokenizer.convert_tokens_to_ids(example['tokens']),\
                    example['span_1'], example['span_2']


            assert len(ids) <= self.seq_len - 2
            
            ids = [self.cls] + ids + [self.sep]

            pad_length = self.seq_len - len(ids)

            attention_mask = (len(ids) * [1]) + (pad_length * [0])

            ids = ids + (pad_length * [self.pad])

                            
            # == [span_1] [span_2] ==
            # primary <-> symbol : direct
            # primary <-> symbol : count
            # primary <-> primary : corefer-to
            # symbol <-> symbol : corefer-to
            # else : None
            
            
            span_1, span_2 = [span_1[0] + 1, span_1[1] + 1], [span_2[0] + 1, span_2[1] + 1]
            
            if mode=='train' or mode=='dev':
                self.dataset.append({"input_ids": ids, 'attention_mask': attention_mask, "labels": int(label_map[example['label']]),
                                "span_1": span_1, "span_2": span_2})
            elif mode=='infer':
                self.dataset.append({"input_ids": ids, 'attention_mask': attention_mask,
                                        "span_1": span_1, "span_2": span_2,
                                        "doc_num": example['doc_num'], 'e1_id': int(example['e1_id'][1:]), 'e2_id': int(example['e2_id'][1:]) })

            ####################
                    
                    

    def __len__(self):
        # return self.dataset_len
        return len(self.dataset)

    def __getitem__(self, item):
        output = self.dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}

    def get_tokenizer_len(self):
        return len(self.tokenizer)
    
