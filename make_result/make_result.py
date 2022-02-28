import json

ner_result = json.load(open("./ensemble_result.json"), encoding="utf-8")

test_data_1 = json.load(open("./origin/cs.ai-ann2.json"), encoding="utf-8")
test_data_2 = json.load(open("./origin/cs.ai-ann3.json"), encoding="utf-8")
test_data_3 = json.load(open("./origin/econ.th-ann4.json"), encoding="utf-8")
test_data_4 = json.load(open("./origin/physics.atom_ph-ann8.json"), encoding="utf-8")
test_data_5 = json.load(open("./origin/q_bio.qm-ann11.json"), encoding="utf-8")

print(ner_result)

for name in test_data_1:
    test_data_1[name] = ner_result[name]
for name in test_data_2:
    test_data_2[name] = ner_result[name]
for name in test_data_3:
    test_data_3[name] = ner_result[name]
for name in test_data_4:
    test_data_4[name] = ner_result[name]
for name in test_data_5:
    test_data_5[name] = ner_result[name]
    
with open('./result/cs.ai-ann2.json', 'w', encoding='utf-8') as make_file:
    json.dump(test_data_1, make_file, indent="\t")    
with open('./result/cs.ai-ann3.json', 'w', encoding='utf-8') as make_file:
    json.dump(test_data_2, make_file, indent="\t")
with open('./result/econ.th-ann4.json', 'w', encoding='utf-8') as make_file:
    json.dump(test_data_3, make_file, indent="\t")
with open('./result/physics.atom_ph-ann8.json', 'w', encoding='utf-8') as make_file:
    json.dump(test_data_4, make_file, indent="\t")
with open('./result/q_bio.qm-ann11.json', 'w', encoding='utf-8') as make_file:
    json.dump(test_data_5, make_file, indent="\t")