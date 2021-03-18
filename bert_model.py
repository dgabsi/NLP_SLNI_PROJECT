import transformers
import torch
import torch.nn as nn



class BertTransformer(nn.Module):
    def __init__(self):

        super().__init__()

        self.bert_model_backbone = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, output_attentions=False, output_hidden_states=False)
        #self.gru=nn.GRU()

    def forward(self, inputs_ids, attention_mask, token_type_ids):

        #print(inputs)
        #print(token_type_ids)
        #print(attention_mask)
        output=self.bert_model_backbone(input_ids=inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        return output