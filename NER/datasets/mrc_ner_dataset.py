#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_dataset.py

import json
import torch
from torch.utils.data import Dataset
from datasets.prepro_ner import prepro_ner, prepro_ner_infer
import tqdm

class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer, max_length: int = 512, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False, mode = 'train', data_dir=""):
        self.tokenizer = tokenizer
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.examples = prepro_ner(docs=self.all_data, tokenizer=self.tokenizer, maxlen=max_length, mode='train')
        self.mode = mode

        label = {'SYMBOL': 0, 'PRIMARY': 1, 'ORDERED': 2}
        label_query_id = {'SYMBOL': self.tokenizer.convert_tokens_to_ids('symbol'),
                          'PRIMARY': self.tokenizer.convert_tokens_to_ids('description'),
                          'ORDERED': self.tokenizer.convert_tokens_to_ids('ordered')}

        self.bos_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.eos_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.tokenizer.pad_token_id

        self.dataset = []
        
        for example in tqdm.tqdm(self.examples): # 각 예제 텍스트길이는 maxlen보다 길수있음. 처리필요.
            # 우선 example tokens, starts, ends, spans 윈도우 단위로 분절하기.
            sliding_length = 100
            maxlen = 500
            
            all_tokens_len = len(example['tokens'])
            boundary_start = maxlen - sliding_length
            boundary_list = []
            
            if all_tokens_len < maxlen:
                boundary_list.append([0, all_tokens_len])
            else:
                boundary_list.append([0, maxlen])
                while all_tokens_len - (boundary_start + 1) > 0:
                    if boundary_start + maxlen < all_tokens_len:
                        boundary_list.append([boundary_start, boundary_start + maxlen])
                        boundary_start += (maxlen - sliding_length)
                    else:
                        boundary_list.append([boundary_start, all_tokens_len])
                        break
            
            example_label = example['label']
            for boundary in boundary_list:
                tokens = example['tokens'][boundary[0]:boundary[1]]
                starts = []
                ends = []
                spans = []
                for span in example['spans']:
                    if boundary[0] <= span[0] and span[1] < boundary[1]:
                        start = span[0]-boundary[0]
                        end = span[1]-boundary[0]
                        starts.append(start)
                        ends.append(end)
                        spans.append([start, end])

                
                query_ids = [label_query_id[example_label]]
                tokens_len = len(tokens)
                query_len = len(query_ids)
                pad_len = max_length - (query_len + tokens_len + 3)

                input_ids = [self.bos_id] + query_ids + [self.eos_id] + \
                            self.tokenizer.convert_tokens_to_ids(tokens) \
                            + [self.eos_id] + ([self.pad_id] * pad_len)
                attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len
                token_type_ids = [0] * (query_len + 2) + [1] * (len(input_ids) - (query_len + 2))

                start_labels = torch.zeros([len(input_ids)], dtype=torch.long)
                for pos in starts:  start_labels[pos + 3] = 1

                end_labels = torch.zeros([len(input_ids)], dtype=torch.long)
                for pos in ends:  end_labels[pos + 3] = 1

                matching_labels = torch.zeros([len(input_ids), len(input_ids)], dtype=torch.long)
                for span in spans:  matching_labels[span[0] + 3][span[1] + 3] = 1
                
                if len(spans) > 0:
                    # print(f"boundary_list: {boundary_list}, all_tokens_len: {all_tokens_len}")
                    # print(spans)
                    # print(len(tokens))
                    
                    self.dataset.append({"input_ids": input_ids,
                                        'attention_mask': attention_mask,
                                        'token_type_ids': token_type_ids,
                                        'start_labels': start_labels,
                                        'end_labels': end_labels,
                                        'matching_labels': matching_labels,
                                        "label": label[example_label],
                                        'doc_id': example['doc_id'],
                                        'all_tok_to_char_list': example['all_tok_to_char_list']
                                        })
            
            # query_ids = [label_query_id[example['label']]]
            # tokens_len = len(example['tokens'])
            # query_len = len(query_ids)
            # pad_len = max_length - (query_len + tokens_len + 3)

            # input_ids = [self.bos_id] + query_ids + [self.eos_id] + \
            #             self.tokenizer.convert_tokens_to_ids(example['tokens']) \
            #             + [self.eos_id] + ([self.pad_id] * pad_len)
            # attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len
            # token_type_ids = [0] * (query_len + 2) + [1] * (len(input_ids) - (query_len + 2))

            # start_labels = torch.zeros([len(input_ids)], dtype=torch.long)
            # for pos in example['starts']:  start_labels[pos + 3] = 1

            # end_labels = torch.zeros([len(input_ids)], dtype=torch.long)
            # for pos in example['ends']:  end_labels[pos + 3] = 1

            # matching_labels = torch.zeros([len(input_ids), len(input_ids)], dtype=torch.long)
            # for span in example['spans']:  matching_labels[span[0] + 3][span[1] + 3] = 1

            # self.dataset.append({"input_ids": input_ids,
            #                     'attention_mask': attention_mask,
            #                     'token_type_ids': token_type_ids,
            #                     'start_labels': start_labels,
            #                     'end_labels': end_labels,
            #                     'matching_labels': matching_labels,
            #                     "label": label[example['label']],
            #                     'doc_id': example['doc_id'],
            #                     'all_tok_to_char_list': example['all_tok_to_char_list']
            #                     })

        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id - 1
        """
        data = self.dataset[item]
        # tokenizer = self.tokenizer

        label_mask = [
            (0 if data['token_type_ids'][token_idx] == 0 else 1)
            for token_idx in range(len(data['input_ids']))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()
        
        return [
            torch.LongTensor(data['input_ids']),
            torch.LongTensor(data['token_type_ids']),
            data['start_labels'],
            data['end_labels'],
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            data['matching_labels'],
            torch.LongTensor([1]), # sample_idx
            torch.LongTensor([1]), #label_idx,
        ]

class MRCNERDataset_infer(Dataset):
    def __init__(self, json_path, tokenizer, max_length: int = 512, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False, mode = 'train', data_dir=""):
        self.tokenizer = tokenizer
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.examples = prepro_ner_infer(docs=self.all_data, tokenizer=self.tokenizer, maxlen=max_length, mode='infer')
        self.mode = mode


        label = {'SYMBOL': 0, 'PRIMARY': 1, 'ORDERED': 2}
        label_query_id = {'SYMBOL': self.tokenizer.convert_tokens_to_ids('symbol'),
                          'PRIMARY': self.tokenizer.convert_tokens_to_ids('description'),
                          'ORDERED': self.tokenizer.convert_tokens_to_ids('ordered')}

        self.bos_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.eos_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.tokenizer.pad_token_id

        self.dataset = []
        
        for example in self.examples:
            sliding_length = 100
            maxlen = 500
            
            all_tokens_len = len(example['tokens'])
            boundary_start = maxlen - sliding_length
            boundary_list = []
            
            if all_tokens_len < maxlen:
                boundary_list.append([0, all_tokens_len])
            else:
                boundary_list.append([0, maxlen])
                while all_tokens_len - (boundary_start + 1) > 0:
                    if boundary_start + maxlen < all_tokens_len:
                        boundary_list.append([boundary_start, boundary_start + maxlen])
                        boundary_start += (maxlen - sliding_length)
                    else:
                        boundary_list.append([boundary_start, all_tokens_len])
                        break
            
            
            for boundary in boundary_list:
                tokens = example['tokens'][boundary[0]:boundary[1]]                            
                for query_ids in [[label_query_id['SYMBOL']], [label_query_id['PRIMARY']]]:
                    
                    # query_ids = [label_query_id[example['label']]]
                    tokens_len = len(tokens)
                    query_len = len(query_ids)
                    pad_len = max_length - (query_len + tokens_len + 3)

                    input_ids = [self.bos_id] + query_ids + [self.eos_id] + \
                                self.tokenizer.convert_tokens_to_ids(tokens) \
                                + [self.eos_id] + ([self.pad_id] * pad_len)
                    # attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len
                    token_type_ids = [0] * (query_len + 2) + [1] * (len(input_ids) - (query_len + 2))
                    
                    assert len(input_ids) == len(token_type_ids)

                    self.dataset.append({"input_ids": input_ids,
                                        'token_type_ids': token_type_ids,
                                        "label": query_ids,
                                        'doc_id': example['doc_id'],
                                        'start_at_doc': boundary[0],
                                        'all_tok_to_char_list': example['all_tok_to_char_list']
                                        })

        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id - 1
        """
        data = self.dataset[item]
        # tokenizer = self.tokenizer

        label_mask = [
            (0 if data['token_type_ids'][token_idx] == 0 else 1)
            for token_idx in range(len(data['input_ids']))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()
        
        
        return [
            torch.LongTensor(data['input_ids']),
            torch.LongTensor(data['token_type_ids']),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            data['doc_id'],
            data['label'],
            data['start_at_doc'],
            data['all_tok_to_char_list']
        ]            