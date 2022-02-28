import json

data_name_list = ['test/cs.ai-ann2.json', 'test/cs.ai-ann3.json', 'test/econ.th-ann4.json',
                  'test/physics.atom_ph-ann8.json', 'test/q_bio.qm-ann11.json'
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

json.dump(all_data, open('test.json', 'w', encoding="utf-8"), indent='\t')

print(len(all_data))
