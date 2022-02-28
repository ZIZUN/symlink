def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def is_symbol(c):
    if c == "\\" or c == "{" or c == "}" or c == "$"or c == "-" \
        or c == "+" or c == "_" or c == "^"or c == "," or c == "=" \
        or c == "[" or c == "]" or c == "(" or c == ")" or c == "<" or c == "*" \
        or c == ">" or c == "|"or c == "/" or c == ":" or c.isdigit() or c.isupper() \
        or c == "." or c == "'" or c == "~" or c == ";"or c == "&":
        return True
    return False

def improve_answer_span(tokens, input_start, input_end, tokenizer, orig_answer_text):  
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def improve_answer_span_2(tokens, input_start, input_end, tokenizer, orig_answer_text):
    orginal_answer_tokens = [] #   ex) ['\','mathrm','{','W','rite','}']
    prev_is_symbol = True

    for c in orig_answer_text:
        if is_symbol(c):
            orginal_answer_tokens.append(c)
            prev_is_symbol = True
            continue
        else:
            if prev_is_symbol:
                orginal_answer_tokens.append(c)
                prev_is_symbol = False
            else:
                orginal_answer_tokens[-1] += c

    a = [tokenizer.tokenize(text) for text in orginal_answer_tokens]
    d = []
    for li in a:
        for i in li:
            d.append(i)
    
    tok_answer_text = " ".join(d)
    
    # tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def prepro_ner(docs, tokenizer, maxlen, mode='train'):
    """ 
    Description:
        tokenizing하고, token <-> char 매핑 리스트 만들어주는 전처리함수.
        reference - Squad 1.0 preprocess code

    Args:
        docs ([type: dict]): dataset
        tokenizer ([type]): Language Model tokenizer
        maxlen ([type: int]): Max length

    Returns:
        example_list([type: list]): [{'tokens' : tokens,
                                 'starts': starts,
                                 'ends': ends,
                                 'spans': spans,
                                 'label': label,
                                 'doc_id': doc_id,
                                 'tok_to_orig_index': tok_to_orig_index,
                                 'tok_to_orig_list': tok_to_orig_list,
                                 'word_to_char_offset': word_to_char_offset,
                                 'context': context
                                 }, ...]
    """
    examples = []
    ne_po = [0,0]
    for doc_name in docs:
        context = docs[doc_name]['text']

        example_for_each_label = {'SYMBOL':[], 'PRIMARY':[], 'ORDERED':[]}

        for eid in docs[doc_name]['entity']:
            entity = docs[doc_name]['entity'][eid]

            entity_label = entity['label']
            orig_answer_text = entity['text']

            context_tokens = []
            char_to_word_offset = []

            prev_is_whitespace = True
            prev_is_symbol = True
            
# "\\Input{Path Probs. 
            for c in context:
                if is_symbol(c):
                    context_tokens.append(c)
                    char_to_word_offset.append(len(context_tokens) - 1)
                    prev_is_symbol = True
                    continue
                
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace or prev_is_symbol:
                        context_tokens.append(c)
                    else:
                        context_tokens[-1] += c
                    prev_is_whitespace = False
                    prev_is_symbol = False
                char_to_word_offset.append(len(context_tokens) - 1)

            assert len(char_to_word_offset) == len(context)
            
            word_to_char_offset = []

            char_span = [0,0]
            temp = 0
            for count, word_offset in enumerate(char_to_word_offset):
                if temp == word_offset:
                    continue
                elif temp != word_offset:
                    char_span[1] = count - 1
                    word_to_char_offset.append(char_span)
                    temp = word_offset
                    char_span = [count,count]
            word_to_char_offset.append([char_span[0], len(char_to_word_offset)])
            
            assert len(word_to_char_offset) == len(context_tokens)


            try:
                start_position = char_to_word_offset[entity['start']]
                end_position = char_to_word_offset[entity['end']]
            except:
                break

            tok_to_orig_index = []
            tok_to_orig_list = []
            orig_to_tok_index = []
            all_context_tokens = []
            all_tok_to_char_list = []
            

            for (i, token) in enumerate(context_tokens):
                orig_to_tok_index.append(len(all_context_tokens))
                sub_tokens = tokenizer.tokenize(token)
                tok_to_char = tokenizer.encode_plus(token, return_offsets_mapping=True)['offset_mapping'][1:-1]
                if i == 0:
                    all_tok_to_char_list+=tok_to_char
                else:
                    plus = word_to_char_offset[i][0]
                    for i, char_tuple in enumerate(tok_to_char):
                        
                        # if is_symbol(context[char_tuple[0]]) and char_tuple[1] - char_tuple[0] == 1:
                        #     tok_to_char[i] = (char_tuple[0]+plus, char_tuple[1]+plus-1)
                        # else:
                        tok_to_char[i] = (char_tuple[0]+plus, char_tuple[1]+plus)
                            
                    
                    all_tok_to_char_list+=tok_to_char
                    
                for (j, sub_token) in enumerate(sub_tokens):
                    tok_to_orig_index.append(i)
                    tok_to_orig_list.append([i,j])
                    all_context_tokens.append(sub_token)

            
            assert len(all_context_tokens) == len(all_tok_to_char_list)

            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(context_tokens) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_context_tokens) - 1
                
                                
            # tok_start_position, tok_end_position = improve_answer_span(all_context_tokens, tok_start_position,
            #                                                            tok_end_position, tokenizer, orig_answer_text)    
            
            tok_start_position, tok_end_position = improve_answer_span_2(all_context_tokens, tok_start_position,
                                                                       tok_end_position, tokenizer, orig_answer_text)



            if context[all_tok_to_char_list[tok_start_position][0]:all_tok_to_char_list[tok_end_position][1]] == 'bending angle is':
                tok_end_position -= 1
            
            a = context[entity['start']:entity['end']]
            b = context[all_tok_to_char_list[tok_start_position][0]:all_tok_to_char_list[tok_end_position][1]]
            
           
            
            # Exclude for minor false train data
            if a != b and (b in ['rr_0}', 'xt\'=', 'nt\'=', '\\mathrm{p^\\alpha_xp', 'tk_{irr}(t\')dt', 'D_xt', 'nt_d\\']):
                continue

            
            # If there is False train data after improve answer span function.
            # Exclude that(ex- answer:d  answer_span: md} ) at train mode
            if a != b and len(a)==1: 
                continue
            
            # if a != b:
            #     print(a,b)
            
            
            if mode=='train':
                example_for_each_label[entity_label].append({
                                                                'doc_id': doc_name,
                                                                'tok_to_orig_index': tok_to_orig_index,
                                                                'tok_to_orig_list': tok_to_orig_list,
                                                                'word_to_char_offset': word_to_char_offset,
                                                                'context': context,
                                                                'orig_answer_text': orig_answer_text,
                                                                'all_context_tokens' : all_context_tokens,
                                                                'all_tok_to_char_list' : all_tok_to_char_list,
                                                                'tok_start_position': tok_start_position,
                                                                'tok_end_position': tok_end_position,
                                                                })
            # elif mode=='infer' and (doc_name not in doc_name_list):
            #     doc_name_list.append(doc_name)
            #     examples.append({'doc_id': doc_name,
            #                     'tokens' : all_context_tokens,
            #                     'context': context,
            #                     'all_tok_to_char_list': all_tok_to_char_list
            #                     })               
                
                
        if mode=='train':
            ### add example for each label
            for label in ['SYMBOL', 'PRIMARY', 'ORDERED']:
                # symbol
                tokens = None
                doc_id = None
                tok_to_orig_index = None
                context = None
                starts = []
                ends = []
                spans = []
                
                for answer in example_for_each_label[label]:
                    if tokens == None:
                        context = answer['context']
                        tokens = answer['all_context_tokens']
                        doc_id = answer['doc_id']
                        tok_to_orig_index = answer['tok_to_orig_index']
                        word_to_char_offset = answer['word_to_char_offset']
                        all_tok_to_char_list = answer['all_tok_to_char_list']
                    starts.append(answer['tok_start_position'])
                    ends.append(answer['tok_end_position'])
                    spans.append([answer['tok_start_position'], answer['tok_end_position']])

                if len(starts) > 0:
                    examples.append({'tokens' : tokens,
                                    'starts': starts,
                                    'ends': ends,
                                    'spans': spans,
                                    'label': label,
                                    'doc_id': doc_id,
                                    'tok_to_orig_index': tok_to_orig_index,
                                    'tok_to_orig_list': tok_to_orig_list,
                                    'word_to_char_offset': word_to_char_offset,
                                    'context': context,
                                    'all_tok_to_char_list': all_tok_to_char_list
                                    })
         
    return examples



def prepro_ner_infer(docs, tokenizer, maxlen, mode='train'):
    """ 
    Description:
        tokenizing하고, token <-> char 매핑 리스트 만들어주는 전처리함수.
        reference - Squad 1.0 preprocess

    Args:
        docs ([type: dict]): dataset
        tokenizer ([type]): Language Model tokenizer
        maxlen ([type: int]): Max length

    Returns:
        example_list([type: list]): [{'tokens' : tokens,
                                 'starts': starts,
                                 'ends': ends,
                                 'spans': spans,
                                 'label': label,
                                 'doc_id': doc_id,
                                 'tok_to_orig_index': tok_to_orig_index,
                                 'tok_to_orig_list': tok_to_orig_list,
                                 'word_to_char_offset': word_to_char_offset,
                                 'context': context
                                 }, ...]
    """
    examples = []
    ne_po = [0,0]
    for doc_name in docs:
        context = docs[doc_name]['text']

        context_tokens = []
        char_to_word_offset = []

        prev_is_whitespace = True
        prev_is_symbol = False


        for c in context:
            if is_symbol(c):
                context_tokens.append(c)
                char_to_word_offset.append(len(context_tokens) - 1)
                prev_is_symbol = True
                continue
            
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace or prev_is_symbol:
                    context_tokens.append(c)
                else:
                    context_tokens[-1] += c
                prev_is_whitespace = False
                prev_is_symbol = False
            char_to_word_offset.append(len(context_tokens) - 1)

        assert len(char_to_word_offset) == len(context)
        
        word_to_char_offset = []

        char_span = [0,0]
        temp = 0
        for count, word_offset in enumerate(char_to_word_offset):
            if temp == word_offset:
                continue
            elif temp != word_offset:
                char_span[1] = count - 1
                word_to_char_offset.append(char_span)                    
                temp = word_offset
                char_span = [count,count]
        word_to_char_offset.append([char_span[0], len(char_to_word_offset)])
        
        assert len(word_to_char_offset) == len(context_tokens)

        all_context_tokens = []
        all_tok_to_char_list = []
        

        for (i, token) in enumerate(context_tokens):
            sub_tokens = tokenizer.tokenize(token)
            tok_to_char = tokenizer.encode_plus(token, return_offsets_mapping=True)['offset_mapping'][1:-1]
            if i == 0:
                all_tok_to_char_list+=tok_to_char
            else:
                plus = word_to_char_offset[i][0]
                for i, char_tuple in enumerate(tok_to_char):
                    tok_to_char[i] = (char_tuple[0]+plus, char_tuple[1]+plus)         
                
                all_tok_to_char_list+=tok_to_char
                
            for (j, sub_token) in enumerate(sub_tokens):
                all_context_tokens.append(sub_token)
        
        assert len(all_context_tokens) == len(all_tok_to_char_list)
        
        doc_name_list = []

        if doc_name not in doc_name_list:
            doc_name_list.append(doc_name)
            examples.append({'doc_id': doc_name,
                            'tokens' : all_context_tokens,
                            'context': context,
                            'all_tok_to_char_list': all_tok_to_char_list
                            })               

    return examples