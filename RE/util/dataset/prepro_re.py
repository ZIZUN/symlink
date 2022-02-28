def prepro_re(docs, tokenizer, maxlen, mode='train'):
    examples = []
    negative_examples = []
    negative_examples_count = 0
    negative_p_p_count = 0
    negative_s_s_count = 0
    negative_p_s_count = 0

    import tqdm
    
    ent_bound_map = {'SYMBOL': ['<S>', '</S>'], 'PRIMARY': ['<P>', '</P>']}
    
    for doc_num, doc_id in tqdm.tqdm(enumerate(docs)):
        context = docs[doc_id]['text']

        example_for_each_label = {}

        for eid in docs[doc_id]['entity']:
            entity = docs[doc_id]['entity'][eid]

            example_for_each_label[entity['eid']]= {'orig_answer_text' : entity['text'],
                                                            'start' : entity['start'],
                                                            'end': entity['end'],
                                                            'label': entity['label'],
                                                            }            
        ent_relation_li = []
        for rid in docs[doc_id]['relation']:
            e1_temp = docs[doc_id]['relation'][rid]['arg0']
            e2_temp = docs[doc_id]['relation'][rid]['arg1']
            ent_relation_li.append((e1_temp, e2_temp))
            
        if mode=='train':
            neg_max = 20000
        elif mode=='dev':
            neg_max = 1000
                

        if mode=='train' or mode=='dev':
            # Make negative examples by Distant supervision
            for i, e1 in enumerate(example_for_each_label):
                
                # if negative_examples_count > neg_max: # TODO: fix
                #     break
                # print(negative_examples_count)
                
                for j, e2 in enumerate(example_for_each_label):
                    if i < j:
                        try:
                            span_1 = [example_for_each_label[e1]['start'], example_for_each_label[e1]['end']]
                            span_2 = [example_for_each_label[e2]['start'], example_for_each_label[e2]['end']]
                        except:
                            continue
                        
                        flag = True
                                                
                        # exclude true positive data
                        for e1_temp, e2_temp in ent_relation_li:
                            if (e1 == e1_temp and e2 == e2_temp) or (e1 == e2_temp and e2 == e1_temp):
                                flag = False
                                break
                            
                        if flag == False:
                            continue            
                        
                        # exclude nested span 1, 2 and 'ORDERED' labeled data.
                        span_1_element = [i for i in range(span_1[0], span_1[1])]
                        span_2_element = [i for i in range(span_2[0], span_2[1])]
                        
                        span_1_label = example_for_each_label[e1]['label']
                        span_2_label = example_for_each_label[e2]['label']

                        for element in span_1_element: 
                            if element in span_2_element:
                                flag = False
                                                    
                        if flag == False or span_1_label == 'ORDERED' or span_2_label == 'ORDERED':
                            continue
                        
                        if span_1[0] >= span_2[0]:   # if order is reversed: span2 -> span1
                            span_1, span_2 = span_2, span_1
                            span_1_label, span_2_label = span_2_label, span_1_label            
                            
                        span_1_left_tokens = tokenizer.tokenize(context[:span_1[0]])
                        span_1_tokens = tokenizer.tokenize(context[span_1[0]:span_1[1]])
                        mid_tokens = tokenizer.tokenize(context[span_1[1]:span_2[0]])
                        span_2_tokens = tokenizer.tokenize(context[span_2[0]:span_2[1]])
                        span_2_right_tokens = tokenizer.tokenize(context[span_2[1]:])
                        
                        all_tokens = span_1_left_tokens + \
                                    [ent_bound_map[span_1_label][0]] + span_1_tokens + [ent_bound_map[span_1_label][1]] \
                                    + mid_tokens + \
                                    [ent_bound_map[span_2_label][0]] + span_2_tokens + [ent_bound_map[span_2_label][1]] \
                                    + span_2_right_tokens
                        span_1 = [len(span_1_left_tokens) + 1, len(span_1_left_tokens) + 1 + len(span_1_tokens)]
                        span_2 = [span_1[1] + 2 + len(mid_tokens), span_1[1] + 2 + len(mid_tokens + span_2_tokens)]
                        
                        
                        if len(all_tokens) <= maxlen - 2:
                            example = {
                                            'tokens' : all_tokens,
                                            'span_1': span_1,
                                            'span_2': span_2,
                                            'e_1_label': span_1_label,
                                            'e_2_label': span_2_label,
                                            'label': 'Negative_Sample'
                                        }
                            negative_examples.append(example)
                            # if span_1_label == span_2_label and span_1_label=='PRIMARY' and negative_p_p_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_p_p_count += 1
                            # elif span_1_label == span_2_label and span_1_label=='SYMBOL' and negative_s_s_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_s_s_count += 1 
                            # elif span_1_label != span_2_label and negative_p_s_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_p_s_count += 1
                            # else:
                            #     continue
                            # negative_examples_count += 1  
                            # print(example)                              
                        else:
                            start = span_1[0]
                            end = span_2[1]
                            
                            if end-start > maxlen-2:
                                continue
                            
                            while len(all_tokens[start:end]) <= maxlen - 4:
                                if start > 0:
                                    start -= 1
                                if end < len(all_tokens):
                                    end += 1
                                    
                            all_tokens = all_tokens[start:end]
                            span_1 = [span_1[0]-start, span_1[1]-start]
                            span_2 = [span_2[0]-start, span_2[1]-start]
                            
                            # print(all_tokens[span_1[0]:span_1[1]], all_tokens[span_2[0]:span_2[1]])
                            # print(span_1_tokens, span_2_tokens)                        
                            
                            example = {
                                            'tokens' : all_tokens,
                                            'span_1': span_1,
                                            'span_2': span_2,
                                            'e_1_label': span_1_label,
                                            'e_2_label': span_2_label,
                                            'label': 'Negative_Sample'
                                        }
                            negative_examples.append(example)
                            # if span_1_label == span_2_label and span_1_label=='PRIMARY' and negative_p_p_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_p_p_count += 1
                            # elif span_1_label == span_2_label and span_1_label=='SYMBOL' and negative_s_s_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_s_s_count += 1 
                            # elif span_1_label != span_2_label and negative_p_s_count <= neg_max/3:
                            #     examples.append(example)
                            #     negative_p_s_count += 1
                            # else:
                            #     continue
                            # negative_examples_count += 1  
                            # print(example)                              
            # Add positive samples
            for rid in docs[doc_id]['relation']:
                e1 = docs[doc_id]['relation'][rid]['arg0']
                e2 = docs[doc_id]['relation'][rid]['arg1']
                label = docs[doc_id]['relation'][rid]['label']

                try:
                    span_1 = [example_for_each_label[e1]['start'], example_for_each_label[e1]['end']]
                    span_2 = [example_for_each_label[e2]['start'], example_for_each_label[e2]['end']]
                except:
                    continue
                
                # exclude nested span 1, 2 and 'ORDERED' labeled data.
                span_1_element = [i for i in range(span_1[0], span_1[1])]
                span_2_element = [i for i in range(span_2[0], span_2[1])]
                flag = True
                
                span_1_label = example_for_each_label[e1]['label']
                span_2_label = example_for_each_label[e2]['label']

                for element in span_1_element: 
                    if element in span_2_element:
                        flag = False
                                               
                if flag == False or span_1_label == 'ORDERED' or span_2_label == 'ORDERED':
                    continue
                
                if span_1[0] >= span_2[0]:   # if order is reversed: span2 -> span1
                    span_1, span_2 = span_2, span_1
                    span_1_label, span_2_label = span_2_label, span_1_label
                
                span_1_left_tokens = tokenizer.tokenize(context[:span_1[0]])
                span_1_tokens = tokenizer.tokenize(context[span_1[0]:span_1[1]])
                mid_tokens = tokenizer.tokenize(context[span_1[1]:span_2[0]])
                span_2_tokens = tokenizer.tokenize(context[span_2[0]:span_2[1]])
                span_2_right_tokens = tokenizer.tokenize(context[span_2[1]:])
                
                all_tokens = span_1_left_tokens + \
                            [ent_bound_map[span_1_label][0]] + span_1_tokens + [ent_bound_map[span_1_label][1]] \
                            + mid_tokens + \
                            [ent_bound_map[span_2_label][0]] + span_2_tokens + [ent_bound_map[span_2_label][1]] \
                            + span_2_right_tokens
                span_1 = [len(span_1_left_tokens) + 1, len(span_1_left_tokens) + 1 + len(span_1_tokens)]
                span_2 = [span_1[1] + 2 + len(mid_tokens), span_1[1] + 2 + len(mid_tokens + span_2_tokens)]
                
                
                if len(all_tokens) <= maxlen - 2:
                    examples.append({
                                    'tokens' : all_tokens,
                                    'span_1': span_1,
                                    'span_2': span_2,
                                    'e_1_label': span_1_label,
                                    'e_2_label': span_2_label,
                                    'label': label
                                    })
                else:
                    start = span_1[0]
                    end = span_2[1]
                    
                    if end-start > maxlen-2:
                        continue
                    
                    while len(all_tokens[start:end]) <= maxlen - 4:
                        if start > 0:
                            start -= 1
                        if end < len(all_tokens):
                            end += 1
                            
                    all_tokens = all_tokens[start:end]
                    span_1 = [span_1[0]-start, span_1[1]-start]
                    span_2 = [span_2[0]-start, span_2[1]-start]
                    
                    # print(all_tokens[span_1[0]:span_1[1]], all_tokens[span_2[0]:span_2[1]])
                    # print(span_1_tokens, span_2_tokens)                        
                    
                    examples.append({
                                    'tokens' : all_tokens,
                                    'span_1': span_1,
                                    'span_2': span_2,
                                    'e_1_label': span_1_label,
                                    'e_2_label': span_2_label,
                                    'label': label
                                    })                          
        elif mode=='infer':
            for i, e1 in enumerate(example_for_each_label):
                for j, e2 in enumerate(example_for_each_label):
                    if i < j:
                        try:
                            span_1 = [example_for_each_label[e1]['start'], example_for_each_label[e1]['end']]
                            span_2 = [example_for_each_label[e2]['start'], example_for_each_label[e2]['end']]
                        except:
                            continue
                        
                        # exclude nested span 1, 2 and 'ORDERED' labeled data.
                        span_1_element = [i for i in range(span_1[0], span_1[1])]
                        span_2_element = [i for i in range(span_2[0], span_2[1])]
                        flag = True
                        
                        span_1_label = example_for_each_label[e1]['label']
                        span_2_label = example_for_each_label[e2]['label']

                        for element in span_1_element: 
                            if element in span_2_element:
                                flag = False
                                                    
                        if flag == False or span_1_label == 'ORDERED' or span_2_label == 'ORDERED':
                            continue
                        
                        if span_1[0] >= span_2[0]:   
                            span_1, span_2 = span_2, span_1
                            span_1_label, span_2_label = span_2_label, span_1_label
                        
                        span_1_left_tokens = tokenizer.tokenize(context[:span_1[0]])
                        span_1_tokens = tokenizer.tokenize(context[span_1[0]:span_1[1]])
                        mid_tokens = tokenizer.tokenize(context[span_1[1]:span_2[0]])
                        span_2_tokens = tokenizer.tokenize(context[span_2[0]:span_2[1]])
                        span_2_right_tokens = tokenizer.tokenize(context[span_2[1]:])
                        
                        all_tokens = span_1_left_tokens + \
                                    [ent_bound_map[span_1_label][0]] + span_1_tokens + [ent_bound_map[span_1_label][1]] \
                                    + mid_tokens + \
                                    [ent_bound_map[span_2_label][0]] + span_2_tokens + [ent_bound_map[span_2_label][1]] \
                                    + span_2_right_tokens
                        span_1 = [len(span_1_left_tokens) + 1, len(span_1_left_tokens) + 1 + len(span_1_tokens)]
                        span_2 = [span_1[1] + 2 + len(mid_tokens), span_1[1] + 2 + len(mid_tokens + span_2_tokens)]
                        
                        # print(all_tokens[span_1[0]:span_1[1]], all_tokens[span_2[0]:span_2[1]])
                        # print(span_1_tokens, span_2_tokens)
                        
                        if len(all_tokens) <= maxlen - 2:
                            examples.append({
                                            'doc_num': doc_num,
                                            'e1_id': e1,
                                            'e2_id': e2,
                                            'tokens' : all_tokens,
                                            'span_1': span_1,
                                            'span_2': span_2,
                                            'e_1_label': span_1_label,
                                            'e_2_label': span_2_label
                                            })
                        else:
                            start = span_1[0]
                            end = span_2[1]
                            
                            if end-start > maxlen-2:
                                continue
                            
                            while len(all_tokens[start:end]) <= maxlen - 4:
                                if start > 0:
                                    start -= 1
                                if end < len(all_tokens):
                                    end += 1
                                    
                            all_tokens = all_tokens[start:end]
                            span_1 = [span_1[0]-start, span_1[1]-start]
                            span_2 = [span_2[0]-start, span_2[1]-start]
                            
                            # print(all_tokens[span_1[0]:span_1[1]], all_tokens[span_2[0]:span_2[1]])
                            # print(span_1_tokens, span_2_tokens)                        
                            
                            examples.append({
                                            'doc_num': doc_num,
                                            'e1_id': e1,
                                            'e2_id': e2,
                                            'tokens' : all_tokens,
                                            'span_1': span_1,
                                            'span_2': span_2,
                                            'e_1_label': span_1_label,
                                            'e_2_label': span_2_label
                                            })              
    if mode!='infer':
        import random
        random.shuffle(negative_examples)
        print(len(negative_examples))
        print(len(examples))
        if mode == "train":
            iter = int(len(negative_examples) / len(examples))
            examples = examples * iter 
            examples += negative_examples #[:len(examples)]
        elif mode == 'dev':
            iter = int(len(negative_examples) / len(examples))
            examples = examples * iter 
            examples += negative_examples
            
            # examples += negative_examples[:len(examples)]
    
    return examples