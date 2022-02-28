from transformers import BertModel, BertConfig, DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch

class RE_classifier(nn.Module):
    def __init__(self, resize_token_embd_len, model_name="base"):
        super().__init__()
        if model_name == 'scibert':
            model_config = BertConfig.from_pretrained(
                pretrained_model_name_or_path='./pretrained_model/scibert_cased/', num_labels=5)
            self.model = BertModel.from_pretrained('./pretrained_model/scibert_cased/', config=model_config)
            self.out_proj = nn.Linear(768 * 3, 5)
        elif model_name == 'scibert_uncased':
            model_config = BertConfig.from_pretrained(
                pretrained_model_name_or_path='allenai/scibert_scivocab_uncased', num_labels=5)
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', config=model_config)
            self.out_proj = nn.Linear(768 * 3, 5)        
        elif model_name=='deberta_v3_large':
            self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')#, config=model_config)           
            self.out_proj = nn.Linear(1024 * 3, 5)
        elif model_name=='deberta_v2_xxlarge':
            self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xxlarge')#, config=model_config)           
            self.out_proj = nn.Linear(1536 * 3, 5)  
        elif model_name=='deberta_v2_xlarge':
            self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')#, config=model_config)           
            self.out_proj = nn.Linear(1536 * 3, 5)                   
                                
        self.model.resize_token_embeddings(resize_token_embd_len)
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss()

    def get_span_representation(self, output, span):
        repre = torch.stack([
            torch.sum(output[i, l[0]:l[1], :], dim=0) / (l[1]-l[0]) for i, l in enumerate(span)
        ])
        return repre
    def forward(self, input_ids, attention_mask,  span_1, span_2, labels=None, inputs_embeds=None):

        outputs = self.model(input_ids, attention_mask, inputs_embeds=inputs_embeds)
        output = outputs.last_hidden_state

        span_1 = self.get_span_representation(output, span_1) #  bsz, hidden
        span_2 = self.get_span_representation(output, span_2) #  bsz, hidden
        cls = output[:,0,:].squeeze(1)

        x = self.dropout(torch.cat([span_1,span_2,cls],dim=1))

        if labels is None:
            return self.out_proj(x)

        logits = self.out_proj(x)

        loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(logits= logits, loss = loss)


