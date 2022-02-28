from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer

all_tok_to_char = []

# tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_cased")
# tokenizer.add_special_tokens({'additional_special_tokens': ['[unused10]']})
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
cls =tokenizer.convert_tokens_to_ids('[CLS]')
sep = tokenizer.convert_tokens_to_ids('[SEP]')
pad = tokenizer.pad_token_id

print(cls,sep,pad)