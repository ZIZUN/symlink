import json
from collections import Counter




if __name__ == '__main__':
    result_file_list = []
    answers_list = []
    
    # result_file_list.append(json.load(open('./result/result1.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result2.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result3.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result4.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result5.json', encoding="utf-8")))
    
    # result_file_list.append(json.load(open('./result/result6.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result7.json', encoding="utf-8")))
    # result_file_list.append(json.load(open('./result/result8.json', encoding="utf-8")))
    result_file_list.append(json.load(open('./result/result_none.json', encoding="utf-8")))
    
    
    model_len = len(result_file_list)
    
    for result_file in result_file_list:
        answers = []
        for doc_name in result_file:
            rel_ans = []
            rels = result_file[doc_name]['relation']
            for rel_name in result_file[doc_name]['relation']:
                e1 = rels[rel_name]['arg0']
                e2 = rels[rel_name]['arg1']
                rel = rels[rel_name]['label']
                
                rel_str = rel+'_'+e1+'_'+e2               
                rel_ans.append(rel_str)

            answers.append(rel_ans)
        answers_list.append(answers)
        
    final_re_result = []
        
    for i in range(300):
        answer_sum=[]
        for answers in answers_list:
            answer_sum += answers[i]
        counter = Counter(answer_sum)
        
        re_result_li = []
        
        for r in counter:
            if counter[r] > 0:#model_len / 2:
                re_result_li.append(r)

        
        ans_dict = {}
        for i, rel in enumerate(re_result_li):
            ans_rel = rel.split('_')
            ans_dict['R'+str(i+1)] = {'label':ans_rel[0], 'arg0':ans_rel[1], 'arg1':ans_rel[2], 'rid':'R'+str(i+1)} 
        final_re_result.append(ans_dict)
            
    
    ensemble_result = json.load(open('./result/result1.json', encoding="utf-8"))
    
    for i, doc_name in enumerate(ensemble_result):
        rel_ans = []
        ensemble_result[doc_name]['relation'] = final_re_result[i]
        
        
    def exclude_not_related_entities(result):
        result_temp = result
        
        for doc_name in result_temp:
            rels = result_temp[doc_name]['relation']
            used_entities = []
            for rel_name in result[doc_name]['relation']:
                used_entities.append(rels[rel_name]['arg0'])
                used_entities.append(rels[rel_name]['arg1'])
            
            entities_name_li = []
            
            for ent_name in result_temp[doc_name]['entity']:
                entities_name_li.append(ent_name)
            
            for ent_name in entities_name_li:
                if ent_name not in used_entities:
                    del result[doc_name]['entity'][ent_name]                                   
        return result
    
    def postpro_corefer_relations(result):
        for doc_name in result:
            rel_names = list(result[doc_name]['relation'].keys())
            rels = result[doc_name]['relation']
            
            corefer_sym_li = []
            corefer_desc_li = []
            
            for rel_name in rel_names:
                arg0 = rels[rel_name]['arg0']
                arg1 = rels[rel_name]['arg1']
                
                if rels[rel_name]['label'] == 'Corefer-Symbol':
                    del result[doc_name]['relation'][rel_name]                 

                    temp_flag = False
                    for i, corefer_sym in enumerate(corefer_sym_li):
                        if arg0 in corefer_sym:
                            if arg1 not in corefer_sym:
                                corefer_sym_li[i].append(arg1)
                                
                            temp_flag = True
                            break
                        elif arg1 in corefer_sym:
                            corefer_sym_li[i].append(arg0)
                            temp_flag = True
                            break
                    if temp_flag:
                        continue
                    else:
                        corefer_sym_li.append([arg0, arg1])
                elif rels[rel_name]['label'] == 'Corefer-Description':
                    del result[doc_name]['relation'][rel_name]                 

                    temp_flag = False
                    for i, corefer_sym in enumerate(corefer_desc_li):
                        if arg0 in corefer_sym:
                            if arg1 not in corefer_sym:
                                corefer_desc_li[i].append(arg1)
                                
                            temp_flag = True
                            break
                        elif arg1 in corefer_sym:
                            corefer_desc_li[i].append(arg0)
                            temp_flag = True
                            break
                    if temp_flag:
                        continue
                    else:
                        corefer_desc_li.append([arg0, arg1])                        
                        
            temp_counter = 0
            for corefer_syms in corefer_sym_li:
                iter_num = len(corefer_syms) - 1
                for i in range(iter_num):
                    result[doc_name]['relation']['R50'+str(temp_counter)] = \
                        {"label": "Corefer-Symbol", "arg0": corefer_syms[i], "arg1": corefer_syms[i+1], "rid": 'R50'+str(temp_counter)}
                    temp_counter += 1
            for corefer_desc in corefer_desc_li:
                iter_num = len(corefer_desc) - 1
                for i in range(iter_num):
                    result[doc_name]['relation']['R50'+str(temp_counter)] = \
                        {"label": "Corefer-Description", "arg0": corefer_desc[i], "arg1": corefer_desc[i+1], "rid": 'R50'+str(temp_counter)}
                    temp_counter += 1            
                              
        return result
        

    ensemble_result = exclude_not_related_entities(ensemble_result)
    ensemble_result = postpro_corefer_relations(ensemble_result)
    
    
    with open('./result/none_result.json', 'w', encoding='utf-8') as make_file:
        json.dump(ensemble_result, make_file, indent="\t")

