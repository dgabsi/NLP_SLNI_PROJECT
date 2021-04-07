import transformers
import torch
import torch.nn as nn



class BertTransformer(nn.Module):
    '''
    Bert Model. Based on bert transformer for classification. topped with linear layer (with tanh) and classifier layer with number of classes outputs
    '''
    def __init__(self, num_classes=3, output_hidden_states=False):

        super().__init__()

        self.bert_model_backbone = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes, output_attentions=True, output_hidden_states=False)
        #self.gru=nn.GRU()

    def forward(self, inputs_ids, attention_mask, token_type_ids, return_attention=False):

        #print(inputs)
        #print(token_type_ids)
        #print(attention_mask)
        output=self.bert_model_backbone(input_ids=inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits=output.logits
        attentions=output.attentions

        if return_attention:
            return logits, attentions

        return logits