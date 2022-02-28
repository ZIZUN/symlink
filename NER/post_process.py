import json
from os import lseek

ner_result = json.load(open("mrc-ner.test.ner_result"), encoding="utf-8")

for doc_id in ner_result:
    if ner_result[doc_id]['topic'] == 'q_bio.qm':
        text = ner_result[doc_id]['text']
        for e_id in ner_result[doc_id]['entity']:
            
            if ner_result[doc_id]['entity'][e_id]['label'] != 'SYMBOL':
                continue
            
            s = ner_result[doc_id]['entity'][e_id]['start']
            e = ner_result[doc_id]['entity'][e_id]['end']
                                    
            if text[s-1] == '$':
                ner_result[doc_id]['entity'][e_id]['start'] -= 1
            if text[e] == '$':
                ner_result[doc_id]['entity'][e_id]['end'] += 1
            
            ner_result[doc_id]['entity'][e_id]['text'] = text[ner_result[doc_id]['entity'][e_id]['start']:ner_result[doc_id]['entity'][e_id]['end']]
            
with open('./mrc-ner.test.ner_result_post', 'w', encoding='utf-8') as make_file:
    json.dump(ner_result, make_file, indent="\t")
    
    

# print(ner_result)