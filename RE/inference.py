import torch.nn.functional as F
import torch
import json
from util.dataset import LoadDataset
from torch.utils.data import DataLoader
from util.model import RE_classifier
import tqdm
label_map = ['Count', 'Direct', 'Corefer-Symbol', 'Corefer-Description', 'Negative_Sample']

def evaluation(best_model_path, corpus_path, seq_len=512, batch_size=200, device='cuda:0',num_workers=5, model_name='base'
               , index_to_docid='', all_data=[]):

    test_dataset = LoadDataset(corpus_path=corpus_path, seq_len=seq_len, mode='infer', model_name=model_name)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    model = RE_classifier(resize_token_embd_len=test_dataset.get_tokenizer_len(), model_name=model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    
    model.to(device)
    model.eval()
    predict_list = []
    result_dict = {}
    
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):
            data = {key: value.to(device) for key, value in data.items()}
            # logits = model.forward(**data)
            logits = model.forward(input_ids=data['input_ids'], attention_mask=data['attention_mask'],  span_1=data['span_1'], span_2=data['span_2'])
            
            predict = F.softmax(logits, dim=1).argmax(dim=1)
            predict_list += predict.tolist()

            predicted_doc_num = data['doc_num'].tolist()
            predicted_e1_id = data['e1_id'].tolist()
            predicted_e2_id = data['e2_id'].tolist()
            predicted_label = predict.tolist()
            # print(predict.tolist())
            
                   
            def check_valid_rel(label, arg0, arg1, all_data, doc_id):
                arg0_label = all_data[doc_id]['entity'][arg0]['label']
                arg1_label = all_data[doc_id]['entity'][arg1]['label']
                
                
                if label == 'Count' or label == 'Direct':
                    if arg0_label == arg1_label:
                        return False
                    else:
                        if arg0_label == 'SYMBOL':
                            return 2
                        else:
                            return True
                elif label == 'Corefer-Symbol':
                    if arg0_label == arg1_label and arg0_label == 'SYMBOL':
                        return True
                    else:
                        return False
                elif label == 'Corefer-Description':
                    if arg0_label == arg1_label and arg0_label == 'PRIMARY':
                        return True
                    else:
                        return False
                elif label == 'Negative_Sample':
                    return False            

                            
            for i in range(len(predicted_label)):
                # print(result_dict)
                
                e1 = 'T'+str(predicted_e1_id[i])
                e2 = 'T'+str(predicted_e2_id[i])
                label = label_map[predicted_label[i]]
                
                if predicted_doc_num[i] not in result_dict:
                    result_dict[predicted_doc_num[i]] = []
                    check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                    if check_valid == 2: # change
                        if e1 != e2:
                            e1, e2 = e2, e1
                        result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                    elif check_valid == 1: # True
                        result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})                        
                else:
                    if len(result_dict[predicted_doc_num[i]]) >= 1:
                        flag = True
                        for rel in result_dict[predicted_doc_num[i]]:
                            if (rel['arg0'] == e1 and rel['arg1'] == e2) or (rel['arg1'] == e1 and rel['arg0'] == e2):
                                flag = False
                                break
                        if flag == False:
                            continue
                            
                        check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                        if check_valid == 2: # change
                            if e1 != e2:
                                e1, e2 = e2, e1
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                        elif check_valid == 1: # True
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})                                
                    else:
                        check_valid = check_valid_rel(label, e1, e2, all_data, index_to_docid[predicted_doc_num[i]])
                        if check_valid == 2: # change
                            if e1 != e2:
                                e1, e2 = e2, e1
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})
                        elif check_valid == 1: # True
                            result_dict[predicted_doc_num[i]].append({"label": label, "arg0": e1, "arg1": e2})        
    return result_dict


if __name__ == '__main__':
    

    ## args ##
    device = 'cuda:1'
    best_model_path = 'output/scibert_uncased/none/5200_94.81_1800_scibert_uncased'
    corpus_path = './mrc-ner.test.ner_result_post2'
    result_path = './result/result_none.json'
    model_name = 'scibert_uncased'
    seq_len = 512
    batch_size = 200
    
    ##################
    
    all_data = json.load(open(corpus_path, encoding="utf-8"))
    
    index_to_docid = {}
    
    for index, doc_id in enumerate(all_data):
        index_to_docid[index] = doc_id
    
    result = evaluation(best_model_path= best_model_path, corpus_path=corpus_path,
                                         seq_len=seq_len, batch_size=batch_size, device=device, num_workers=5,
                                         model_name=model_name, index_to_docid=index_to_docid, all_data=all_data)
    for id, doc_id in enumerate(all_data):
        all_data[doc_id]['relation'] = {}
        
        if id not in result:
            continue
        
        for rid, relation in enumerate(result[id]):
            relation['rid'] = 'R'+str(rid+1)
            all_data[doc_id]['relation']['R'+str(rid+1)] = relation


    with open(result_path, 'w', encoding='utf-8') as make_file:
        json.dump(all_data, make_file, indent="\t")

