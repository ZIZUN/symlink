import json
from os import lseek

data_name_list = ['train/cs.ai-ann0.json', 'train/cs.ai-ann2.json', 'train/cs.ai-ann3.json',
                  'train/econ.th-ann4.json', 'train/econ.th-ann5.json', 'train/econ.th-ann6.json',
                  'train/math.co-ann7.json', 'train/physics.atom_ph-ann8.json', 'train/physics.atom_ph-ann9.json',
                  'train/physics.atom_ph-ann10.json', 'train/q_bio.qm-ann11.json',
                  'dev/cs.ai-ann0_dev.json', 'dev/cs.ai-ann2_dev.json', 'dev/econ.th-ann4_dev.json',
                  'dev/math.co-ann7_dev.json', 'dev/physics.atom_ph-ann9_dev.json'
                  ]

import re
# for m in re.finditer('    ', text):
#     text= text[:m.start()] + ' [] ' + text[m.end():]

all_data = {}
train_data = {}
dev_data = {}
for data_name in data_name_list:
    all_data.update(json.load(open(data_name, encoding="utf-8")))

all_name_list = [name for name in all_data]
for name in all_name_list:
    # chage blank to special token
    textlen = len(all_data[name]['text'])

    text = all_data[name]['text'].replace('\n', ' ')
    for m in re.finditer('            ', text):
        text = text[:m.start()] + ' [unused10] ' + text[m.end():]
    all_data[name]['text'] = text

    text = all_data[name]['text']
    for m in re.finditer('            ', text):
        text = text[:m.start()] + ' [unused10] ' + text[m.end():]
    all_data[name]['text'] = text

    text = all_data[name]['text']
    for m in re.finditer('           ', text):
        text = text[:m.start()] + '[unused10] ' + text[m.end():]
    all_data[name]['text'] = text

    text = all_data[name]['text']
    for m in re.finditer('999999[0-9][0-9][0-9][0-9]', text):
        text = text[:m.start()] + '[unused10]' + text[m.end():]
    all_data[name]['text'] = text

    assert textlen == len(all_data[name]['text'])

    entity_name_list = [name for name in all_data[name]['entity']]
    for e_n in entity_name_list:
        s = all_data[name]['entity'][e_n]['start']
        e = all_data[name]['entity'][e_n]['end']

        orig_text = all_data[name]['text'][s:e]
        
        if orig_text.count('$') % 2 == 0 and orig_text[0]== "$" and orig_text[-1]=="$":
            all_data[name]['entity'][e_n]['text'] = orig_text[1:-1]
            all_data[name]['entity'][e_n]['start'] += 1
            all_data[name]['entity'][e_n]['end'] -= 1
        
        
        s = all_data[name]['entity'][e_n]['start']
        e = all_data[name]['entity'][e_n]['end']

        orig_text = all_data[name]['text'][s:e]
        
        if orig_text.count('$') == 1 and orig_text[0]== "$":
            all_data[name]['entity'][e_n]['text'] = orig_text[1:]
            all_data[name]['entity'][e_n]['start'] += 1
        if orig_text.count('$') == 1 and orig_text[-1]== "$":
            all_data[name]['entity'][e_n]['text'] = orig_text[:-1]
            all_data[name]['entity'][e_n]['end'] -= 1
            
            

            

            
        
        entity = all_data[name]['entity'][e_n]
        li = entity['text'].split(' ')
        flag = 0
        for count, word in enumerate(li):
            if word[:6] == "999999" and len(word) == 10:
                flag = 1
                li[count] = '[unused10]'
        if flag == 1:
            processed_entity = ' '.join(li)
            # print(processed_entity)
            all_data[name]['entity'][e_n]['text'] = processed_entity
            # text = text[:entity['start']] + processed_entity + text[entity['end']:]
            all_data[name]['text'] = all_data[name]['text'][:entity['start']] + processed_entity + \
                                     all_data[name]['text'][entity['end']:]


        # print(all_data[name]['text'][s:e])
        # print(all_data[name]['entity'][e_n]['text'])

        # assert all_data[name]['text'][s:e].replace('\n', ' ') == all_data[name]['entity'][e_n]['text']
        if all_data[name]['text'][s:e] != all_data[name]['entity'][e_n]['text']:
            del(all_data[name]['entity'][e_n])
        else:
            assert all_data[name]['text'][s:e] == all_data[name]['entity'][e_n]['text']

            # print('-----')
            # print(all_data[name]['text'][s:e])
            # print(all_data[name])
            # print(all_data[name]['entity'][e_n]['text'])
            # print('-----')
    assert textlen == len(all_data[name]['text'])

    if len(all_data[name]['entity']) == 0:
        dev_data[name] = all_data[name]
    else:
        train_data[name] = all_data[name]
import random
train_name_list = [name for name in train_data]
random.shuffle(train_name_list)
for count, name in enumerate(train_name_list):
    if count == 80:
        break
    else:
        dev_data[name] = train_data[name]
        del(train_data[name])
json.dump(train_data, open('train.json', 'w', encoding="utf-8"), indent='\t')
json.dump(dev_data, open('dev.json', 'w', encoding="utf-8"), indent='\t')

print(len(all_data))
print(len(train_data), len(dev_data))
print(len(train_data) + len(dev_data))